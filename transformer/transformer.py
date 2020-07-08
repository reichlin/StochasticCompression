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
                 num_enc_layers=3,
                 num_dec_layers=3,
                 z_centroids=None,
                 mse_method=False,
                 num_centroids=8
                 ):
        super().__init__()

        self.mse_method = mse_method

        self.d_model = d_model
        self.position_embedding = PositionEmbedding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_dec_layers)

        # The shape is updated in each forward pass to not have to give it as an argument in other functions
        self.z_shape = None  # ~ B x D x H x W
        self.z_centroids = z_centroids
        self.arange = torch.arange(len(z_centroids), dtype=torch.float)

        self._init_parameters()

    def forward(self, z):
        # Encode z to c, then decode to z_tilde which is
        self.z_shape = z.shape  # ~ B x D x H x W

        c = self.encode(z)   # c ~ HW x B x D

        if self.mse_method:
            z_reals, z_logits = self.decode(c)
            z_idx = (self.z_centroids.view(-1, 1, 1, 1, 1) - z_reals.unsqueeze(0)).abs().argmin(dim=0)  # find closest index
            z_rec = self.z_centroids[z_idx].detach()

        else:
            z_reals, z_logits = self.decode(c)
            z_rec = self.z_centroids[z_logits.argmax(dim=1)].detach()  # mask the same way z is masked

        output = {'c': c,
                  'z_rec': z_rec,
                  'z_reals': z_reals,
                  'z_logits': z_logits,
                  }

        return output

    def encode(self, z):
        # z  ~ B x D x H x W
        # c  ~ HW x B x D
        # mu ~ B

        z = self.position_embedding(z)  # B x D x H x W, with pos. emb. added
        z = utils.flatten_tensor_batch(z)  # to transformer format --> HW x B x D

        c = self.encoder(z)

        return c

    def decode(self, c):

        b, d, h, w = self.z_shape

        tgt = torch.zeros((b, d, h, w))
        tgt = self.position_embedding(tgt)
        tgt = utils.flatten_tensor_batch(tgt)

        tgt_mask = utils.generate_subsequent_mask(seq_len=int(h * w))  # generate z_tilde autoregressively

        z_reals = self.decoder(tgt, c, tgt_mask)
        z_reals = utils.unravel_tensor_batch(z_reals, self.z_shape)  # HW x B x D --> B x D x H x W
        z_reals = z_reals.sigmoid() * (len(self.z_centroids) - 1)  # squash to range [0, L]
        # todo: this would be pushed into range of z centroids

        if self.mse_method:
            return z_reals, None
        else:
            # Maps from B x 1 x D x H x W --> B x L x D x H x W
            z_logits = 1 - (self.arange.view(1, -1, 1, 1, 1) - z_reals.unsqueeze(1)).abs()  # alfredos tricks
            return z_reals, z_logits

    def _init_parameters(self):
        """
        Initialize parameters in the transformer model.
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def transformer_loss(z, z_logits, z_centroids):
    """
    z               ~ B x D x H x W, quantized with z_centroids and masked
    z_tilde_logits  ~ B x L x D x H x W, L is the number of centroids
    z_centroids     ~ L, symbol centroids for z
    """

    z_mask = ~(z.abs() > 0)

    z_target = (z.unsqueeze(0) - z_centroids.view(-1, 1, 1, 1, 1)).abs().argmin(dim=0)  # map z to centroid index
    z_target[z_mask] = -1  # set the masked elements to -1 and ignore that in the loss calculation
    loss = F.cross_entropy(z_logits, z_target, ignore_index=-1)

    # If symbols are [0, 1, ..., L-1]
    # z_rec = z_tilde_logits.argmax(dim=1)
    # z_target = z.to(int)
    # loss = F.cross_entropy(z_logits, z_target)

    # If accuracy is calc. in here too..
    # z_rec = z_centroids[z_logits.argmax(dim=1)]
    # equal = (z == z_rec)
    # accuracy = equal.sum(dtype=torch.float) / z.shape.numel()

    return loss  #, accuracy


def mse_loss(z, z_tilde, z_centroids):

    z_reconstructed = (z_centroids.view(-1, 1, 1, 1, 1) - z_tilde.unsqueeze(0)).abs().argmin(dim=0)
    accuracy = (z == z_reconstructed).sum(dtype=torch.float) / z.shape.numel()
    loss = F.mse_loss(z_tilde, z)

    return loss, accuracy


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


def train(model, optimizer, writer, z, z_centroids, EPOCHS):

    mask = (z == 0)

    for epoch in tqdm(range(EPOCHS)):

        output = model(z)

        c = output['c']
        z_rec = output['z_rec']
        z_reals = output['z_reals']
        z_logits = output['z_logits']

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

        if epoch % 10000 == 0:
            print()

        if model.mse_method:
            loss, accuracy = mse_loss(z, z_reals, z_centroids)
        else:
            loss = transformer_loss(z, z_logits, z_centroids)

        # only look at the unmasked positions in z to calc. accuracy
        accuracy = (z == z_rec).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())

        if epoch % 50 == 0:
            print(f'\nLoss: {loss.item():10.2f}\nAccuracy: {accuracy.item():6.3f}')
            # print(f'Range of c: [{c.detach().min():.3f}, {c.detach().max():.3f}]')
            if model.mse_method:
                print(f'Range of z_reals: [{z_reals.detach().min():.3f}, {z_reals.detach().max()}]')
            else:
                print(f'z uniques: ', z_rec.unique())

        if model.mse_method:
            writer.add_scalar('mse_loss', loss.item(), epoch)
        else:
            writer.add_scalar('loss', loss.item(), epoch)
            writer.add_histogram('z_reals_hist', z_reals.detach().flatten(0), epoch)

        writer.add_scalar('accuracy', accuracy.item(), epoch)

        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        # scheduler.step()

    writer.close()


def main():

    torch.manual_seed(1234)

    transform = transforms.Compose([transforms.Resize(168),
                                    transforms.CenterCrop(168),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(root='../Kodak/', transform=transform)
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

    # z, z_centroids = get_fake_z()

    print(f'symbol dist: {z.unique(return_counts=True)[1]}')  # [160, 128, 128,  96, 192,  96, 192, 160]

    # d_ff = 128
    # nhead = 4
    # num_enc_layers = num_dec_layers = 1

    nhead = 4
    d_ff = 2048  # 4096 works even better, at least for overfitting (:
    dropout = 0.0
    num_layers = 6

    num_enc_layers = num_dec_layers = num_layers
    model = CompressionTransformer(model_size, nhead, d_ff, dropout, num_enc_layers,
                                   num_dec_layers, z_centroids, mse_method=False)

    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    model.train()
    model_name = f'nhead_{nhead}_dff_{d_ff}_encdeclayers_{num_enc_layers}_Kodak_first_4'
    writer = SummaryWriter('./runs/' + model_name)
    train(model, optimizer, writer, z, z_centroids, EPOCHS=5000)


if __name__ == '__main__':
    main()