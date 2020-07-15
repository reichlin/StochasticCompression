from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

import torch.optim as optim

import transformer_utils as utils
from network import Net


class CompressionTransformer(nn.Module):

    def __init__(self,
                 d_model=64,
                 nhead=4,
                 d_ff=256,
                 dropout=0.1,
                 num_centroids=8,
                 num_enc_layers=3,
                 num_dec_layers=3,
                 z_centroids=None,
                 pareto_alpha=1.16,
                 pareto_interval=0.3
                 ):
        super().__init__()

        self.d_model = d_model
        self.position_embedding = PositionEmbedding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layers - 1)
        self.mu_encoder = MuEncoderLayer(d_model, nhead, d_ff, dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_dec_layers)

        # The shape is updated in each forward pass to not have to give it as an argument in other functions
        self.z_shape = None  # ~ B x D x H x W
        self.z_centroids = z_centroids
        self.arange = torch.arange(len(z_centroids), dtype=torch.float)

        # Symbol centroids
        initial_centroids = torch.linspace(-4., 4., num_centroids)
        self.c_centroids = torch.nn.Parameter(initial_centroids)  # centroids for symbols in c

        # Pareto constants
        self.pareto_alpha = pareto_alpha
        self.pareto_interval = pareto_interval

        self._init_parameters()

    def forward(self, z, training=True):
        # Encode z to c, then decode to z_rec
        self.z_shape = z.shape  # ~ B x D x H x W

        c, mewtwo = self.encode(z)   # c ~ HW x B x D, mu ~ B

        # compute k for learning optimal mu and k_compress to push information
        if training:
            k, log_pk = self.sample_k(mewtwo)
            k_compress = self.sample_compression(mewtwo)
        else:
            k, log_pk = mewtwo, None
            k_compress = mewtwo

        c = self.quantize(c)

        c_k, memory_mask_k = self.mask(c, k)
        c_compress, memory_mask_compress = self.mask(c, k_compress)

        # z_real, z_logits = self.decode(c)
        z_real_k, z_logits_k = self.decode(c_k)
        z_real_compress, z_logits_compress = self.decode(c_compress)

        # z_rec = self.z_centroids[z_logits.argmax(dim=1)].detach()  # mask the same way z is masked?
        z_rec_k = self.z_centroids[z_logits_k.argmax(dim=1)].detach()  # mask the same way z is masked
        z_rec_compress = self.z_centroids[z_logits_compress.argmax(dim=1)].detach()  # mask the same way z is masked

        output = {'k': k,
                  'k_compress': k_compress,
                  'c_k': c_k,
                  'log_pk': log_pk,
                  'mewtwo': mewtwo,
                  'z_rec_k': z_rec_k,
                  'z_logits_k': z_logits_k,
                  'z_rec_compress': z_rec_compress,
                  'z_logits_compress': z_logits_compress
                  }

        return output

    def encode(self, z):
        # z  ~ B x D x H x W
        # c  ~ HW x B x D
        # mu ~ B

        z = self.position_embedding(z)  # B x D x H x W, with pos. emb. added
        z = utils.flatten_tensor_batch(z)  # to transformer format --> HW x B x D

        c = self.encoder(z)
        c, mewtwo = self.mu_encoder(c)

        return c, mewtwo

    def decode(self, c, memory_mask=None):

        b, d, h, w = self.z_shape

        tgt = torch.zeros((b, d, h, w))
        tgt = self.position_embedding(tgt)
        tgt = utils.flatten_tensor_batch(tgt)

        tgt_mask = utils.generate_subsequent_mask(seq_len=int(h * w))  # generate z_tilde autoregressively

        z_real = self.decoder(tgt, c, tgt_mask, memory_key_padding_mask=memory_mask)
        z_real = utils.unravel_tensor_batch(z_real, self.z_shape)  # HW x B x D --> B x D x H x W
        z_real = z_real.sigmoid() * (len(self.z_centroids) - 1)  # squash to range [0, L-1]

        # Maps from B x 1 x D x H x W --> B x L x D x H x W
        z_logits = 1 - (self.arange.view(1, -1, 1, 1, 1) - z_real.unsqueeze(1)).abs()

        return z_real, z_logits

    def quantize(self, c):
        norm = (torch.abs(c.unsqueeze(-1) - self.c_centroids)) ** 2
        c_tilde = (F.softmax(-1.0 * norm, dim=-1) * self.c_centroids).sum(-1)
        c_hat = (F.softmax(-1e7 * norm, dim=-1) * self.c_centroids).sum(-1)
        c_quantized = (c_hat - c_tilde).detach() + c_tilde  # differentiability trick

        return c_quantized

    def mask(self, c, k):
        # c ~ HW x B x D
        # k ~ B
        # todo: Is detach() needed here? k is already detached in sample_k(mu)..

        k = k.detach()

        b, d, h, w = self.z_shape
        mask_from_index = (k * (d * h * w)).to(int).unsqueeze(1)  # B x 1
        mask = ~torch.arange(d * h * w).repeat(b, 1).ge(mask_from_index)  # B x DHW
        mask = mask.view(b, h * w, d).permute(1, 0, 2)  # HW x B x D, reshaped to format of c
        c_masked = c * mask

        # mask for the transformer decoder to know which elements in the sequence that are masked
        memory_mask = torch.arange(h * w).unsqueeze(0).ge(torch.ceil(mask_from_index / float(d)))  # B x HW
        return c_masked, memory_mask

    def sample_k(self, mu):
        # Poisson sampling
        # mu ~ B

        b, d, h, w = self.z_shape
        n = float(d * h * w)

        m = torch.distributions.Poisson(mu * n)
        k = torch.clamp(m.sample(), 0., n)

        log_pk = m.log_prob(k.detach())
        k = (k / n).detach()

        return k, log_pk

    def sample_compression(self, mu):
        # Pareto sampling
        # mu ~ B

        b, d, h, w = self.z_shape
        n = float(d * h * w)

        delta = mu * self.pareto_interval

        l = torch.clamp(torch.ceil((mu - delta) * n), 1.0, float(n - 3.0))
        h = torch.clamp(torch.ceil((mu + delta) * n), 4.0, float(n))
        alpha = self.pareto_alpha
        k_compression = torch.clamp(self.sample_bounded_pareto(mu.shape, alpha, l, h), 0, n) / n

        return k_compression

    '''
        a = alpha of the bounded pareto -> gives the steepness
        l = lower bound
        h = upper bound
        returns a random sample form this distribution
    '''
    def sample_bounded_pareto(self, size, a, l, h):
        # if self.cuda_backend:
        #     u = torch.cuda.FloatTensor(size).uniform_()
        # else:
        #     u = torch.FloatTensor(size).uniform_()

        u = torch.FloatTensor(size).uniform_()
        num = (u * torch.pow(h, a) - u * torch.pow(l, a) - torch.pow(h, a))
        den = (torch.pow(h, a) * torch.pow(l, a))
        x = torch.pow(- num / den, (-1. / a))
        return x

    def _init_parameters(self):
        """
        Initialize parameters in the transformer model.
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MuEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model + 1)  # an extra channel for computing mu

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model + 1)  # an extra channel for computing mu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2 = self.attention(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # skip connection
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))  # Feedforward
        src = self._pad(src) + self.dropout2(src2)  # skip connection

        # todo: this layer norm potentially mess up by forcing c values to be small
        src = self.norm2(src)

        out = src[:, :, :-1]  # ~ HW x B x D
        mu = src[:, :, -1].mean(dim=0).sigmoid()  # HW x B take mean over HW --> B

        return out, mu

    def _pad(self, x):
        """
        Pad x with zeros in the channel dimension to allow for skip connection
        """
        hw, b, d = x.shape
        return torch.cat((x, torch.zeros(hw, b, 1)), dim=2)


class PositionEmbedding2D(nn.Module):
    """
    Applies 2D positional embedding
    """

    def __init__(self, d_model, max_h=200, max_w=200):
        super().__init__()

        assert d_model % 4 == 0

        x_embed = torch.arange(0, max_w).repeat(1, max_h, 1)
        y_embed = torch.arange(0, max_h).repeat(1, max_w, 1).permute(0, 2, 1)
        dim_t = torch.arange(d_model // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / d_model // 2)
        pos_x = x_embed.unsqueeze(dim=3) / dim_t  # same as x_embed[:, :, :, None] / dim_t
        pos_y = y_embed.unsqueeze(dim=3) / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        self.pos_embedding = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    def forward(self, src):
        b, _, h, w = src.shape
        # Slice the position embedding with the size of the src "image"
        pos_embedding = self.pos_embedding[:, :, :h, :w]

        return src + pos_embedding


class TransformerLoss:

    def __init__(self, z_centroids, threshold=0.98):
        self.threshold = threshold
        self.z_centroids = z_centroids

    def __call__(self, z, z_rec_compress, z_logits_compress, k_2, log_pk_2):

        mask = (z == 0)
        accuracy = (z == z_rec_compress).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())
        policy_loss = (accuracy > self.threshold).detach()

        z_target = (z.unsqueeze(0) - self.z_centroids.view(-1, 1, 1, 1, 1)).abs().argmin(dim=0)
        z_target[mask] = -1  # set the masked elements to -1 and ignore that in the loss calc.
        loss_ce = F.cross_entropy(z_logits_compress, z_target, ignore_index=-1)

        loss_pg = policy_loss * (k_2 * log_pk_2).mean()

        # gamma = 1.5
        # loss = loss_ce + gamma * mewtwo.mean()
        # policy_loss = False

        return loss_ce, loss_pg, policy_loss

    def old__call__(self, z, z_rec_k, z_logits_k, k_2, log_pk_2):

        mask = (z == 0)
        accuracy = (z == z_rec_k).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())
        policy_loss = (accuracy > self.threshold).detach()

        z_target = (z.unsqueeze(0) - self.z_centroids.view(-1, 1, 1, 1, 1)).abs().argmin(dim=0)
        z_target[mask] = -1  # set the masked elements to -1 and ignore that in the loss calc.
        loss_ce = F.cross_entropy(z_logits_k, z_target, ignore_index=-1)

        loss_pg = policy_loss * (k_2 * log_pk_2).mean()

        # gamma = 1.5
        # loss = loss_ce + gamma * mewtwo.mean()
        # policy_loss = False

        return loss_ce, loss_pg, policy_loss


def train(model, optimizer, writer, z, z_centroids, EPOCHS, gamma=0.1, threshold=0.9):

    mask = (z == 0)
    transf_loss = TransformerLoss(z_centroids, threshold=threshold)

    for epoch in tqdm(range(EPOCHS)):

        output = model(z)

        k = output['k']
        k_compress = output['k_compress']
        c_k = output['c_k']
        log_pk = output['log_pk']
        mewtwo = output['mewtwo']
        z_rec_k = output['z_rec_k']
        z_rec_compress = output['z_rec_compress']
        z_logits_k = output['z_logits_k']
        z_logits_compress = output['z_logits_compress']

        # if epoch % 100 == 0:
        #     B, C, H, W = z.shape
        #     acc_symb = np.zeros((8))
        #     num_symbs = np.zeros((8))
        #     for b in range(B):
        #         for c in range(C):
        #             for h in range(H):
        #                 for w in range(W):
        #                     label = int(z[b, c, h, w])
        #                     num_symbs[label] += 1
        #                     acc_symb[label] += float(z[b, c, h, w] == z_rec[b, c, h, w])
        #
        #     acc_symb /= num_symbs
        #     for l in range(8):
        #         writer.add_scalar('Acc_symbol_' + str(l), acc_symb[l], epoch / 100)

        # loss = transformer_loss(z, z_logits, z_centroids)
        loss_ce, loss_pg, policy_loss = transf_loss(z, z_rec_compress, z_logits_compress, k, log_pk)

        loss = loss_ce + gamma * loss_pg

        if policy_loss:
            print(f'PG loss on epoch {epoch}!')

        avg_num_symbols = (c_k != 0).sum(dtype=torch.float).detach() / c_k.shape[1]

        # only look at the unmasked positions in z to calc. accuracy
        accuracy_k = (z == z_rec_k).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())
        accuracy_compress = (z == z_rec_compress).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())

        if epoch % 50 == 0:
            print(f'\nLoss: {loss.item():.3f}')
            print(f'\nLoss ce: {loss_ce.item():.3f}')
            print(f'\nLoss pg: {loss_pg.item():.3f}')
            print(f'Accuracy k: {accuracy_k.item():.3f}')
            print(f'Accuracy compress: {accuracy_compress.item():.3f}')
            print('z_k uniques: ', z_rec_k.unique())
            print('z_compress uniques: ', z_rec_compress.unique())
            print('c_k uniques: ', c_k.detach().unique())
            print('c_compress uniques: ', c_k.detach().unique())
            print('avg num symbols: ', avg_num_symbols)

        if writer is not None:
            writer.add_scalar('k mean', k.mean().item(), epoch)
            writer.add_scalar('k_compress mean', k_compress.mean().item(), epoch)
            writer.add_scalar('mewtwo mean', mewtwo.mean().item(), epoch)
            writer.add_scalar('loss', loss.item(), epoch)
            writer.add_scalar('loss_ce', loss_ce.item(), epoch)
            writer.add_scalar('loss_pg', loss_pg.item(), epoch)
            writer.add_histogram('c_k', c_k.detach().flatten(0), epoch)
            writer.add_scalar('accuracy_k', accuracy_k.item(), epoch)
            writer.add_scalar('accuracy_compress', accuracy_compress.item(), epoch)
            writer.add_scalar('policy_gradient_loss', int(policy_loss), epoch)
            writer.add_scalar('avg_num_symbols', avg_num_symbols, epoch)

        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

    writer.close()

    return model


def main():

    torch.manual_seed(1234)

    transform = transforms.Compose([transforms.Resize(168),
                                    transforms.CenterCrop(168),
                                    transforms.ToTensor()])

    path_to_testset = '../Kodak/'

    dataset = datasets.ImageFolder(root=path_to_testset, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False)

    # take first 4 images
    img, _ = next(iter(loader))

    L = 8
    n = 64
    colors = 3
    model_depth = 3
    model_size = 64
    min_accuracy = 97.0
    pareto_alpha = 1.16
    pareto_interval = 0.5
    cuda_backend = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_backend else "cpu")

    net = Net(min_accuracy, colors, model_depth, model_size, n, L, cuda_backend, pareto_interval, pareto_alpha)
    net.load_state_dict(torch.load('../models/accuracy_97.597206.pt', map_location=device), strict=False)

    out = net(img)
    z = out[-1].detach()
    z_centroids = net.c.detach()

    print(f'symbol dist: {z.unique(return_counts=True)[1]}')  # [160, 128, 128,  96, 192,  96, 192, 160]


    nhead = 4
    d_ff = 2048  # 4096 works even better, at least for overfitting (:
    dropout = 0.0
    num_layers = 6
    num_centroids = 8
    num_enc_layers = num_dec_layers = num_layers

    model = CompressionTransformer(model_size, nhead, d_ff, dropout, num_centroids,
                                   num_enc_layers, num_dec_layers, z_centroids)

    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    model.train()
    model_name = f'pareto_mask_nhead_{nhead}_dff_{d_ff}_encdeclayers_{num_enc_layers}_Kodak_first_4'
    writer = SummaryWriter('./runs/' + model_name)
    # writer = None

    model = train(model, optimizer, writer, z, z_centroids, EPOCHS=5000)

    save_path = './models/' + model_name + '.pt'
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()