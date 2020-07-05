import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import CIFAR10

import transformer_utils as utils


class CompressionTransformer(nn.Module):

    def __init__(self,
                 d_model=64,
                 nhead=4,
                 d_ff=256,
                 dropout=0.1,
                 num_centroids=8,
                 num_enc_layers=3,
                 num_dec_layers=3,
                 z_centroids=None
                 ):
        super().__init__()

        self.d_model = d_model
        self.position_embedding = PositionEmbedding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layers - 1)
        # last transformer encoder that also outputs mu
        self.mu_encoder = MuEncoderLayer(d_model, nhead, d_ff, dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_dec_layers)

        self.arange = torch.arange(num_centroids, dtype=torch.float) / num_centroids

        # The shape is updated in each forward pass to not have to give it as an argument in other functions
        self.z_shape = None  # ~ B x D x H x W

        # Symbol centroids
        initial_centroids = torch.linspace(-num_centroids / 2. * d_model,
                                           num_centroids / 2. * d_model,
                                           num_centroids)
        self.centroids = torch.nn.Parameter(initial_centroids)  # centroids for symbols in c
        self.z_centroids = z_centroids.detach()

        self._init_parameters()

    def forward(self, z):
        # Encode z to c, then decode to z_tilde which is
        self.z_shape = z.shape  # ~ B x D x H x W

        z_mask = z.abs() > 0  # True where we have a symbol, else False (where z is masked)

        c, mu = self.encode(z)   # c ~ HW x B x D, mu ~ B

        k = log_pk = 0
        # k, log_pk = self.sample_k(mu)  # k ~ B, log_pk ~ B
        #
        # k[:] = log_pk[:] = 1  # todo: delete this when reconstruction works..
        #
        # c = self.quantize(c)
        # c, memory_mask = self.mask(c, k)  # Most compressed

        z_tilde_logits = self.decode(c, memory_mask=None)
        z_tilde = self.z_centroids[z_tilde_logits.argmax(dim=1)] * z_mask  # mask the same way z is masked

        output = {'c': c,
                  'mewtwo': mu,
                  'z_tilde': z_tilde,
                  'k_transf': k,
                  'log_pk_transf': log_pk,
                  'z_tilde_logits': z_tilde_logits
                  }

        return output

    def sample_k(self, mu):
        # mu ~ B
        b, d, h, w = self.z_shape
        n = float(d * h * w)

        m = torch.distributions.Poisson(mu * n)
        k = torch.clamp(m.sample(), 0., n)

        log_pk = m.log_prob(k.detach())
        k = (k / n).detach()

        return k, log_pk

    def encode(self, z, padding_mask=None):
        # z  ~ B x D x H x W
        # c  ~ HW x B x D
        # mu ~ B

        z = self.position_embedding(z)  # B x D x H x W, with pos. emb. added
        z = utils.flatten_tensor_batch(z)  # to transformer format --> HW x B x D
        if padding_mask is not None:
            padding_mask = utils.flatten_mask(padding_mask)

        # todo: maybe layernorm before passing z as input, like in DETR

        src = self.encoder(z, src_key_padding_mask=padding_mask)  # first standard encoding layers
        c, mu = self.mu_encoder(src, src_key_padding_mask=padding_mask)  # last encoding layer with mu

        return c, mu

    def decode(self, c, memory_mask=None):

        b, d, h, w = self.z_shape
        tgt_mask = utils.generate_subsequent_mask(seq_len=int(h * w))

        tgt = torch.zeros((h * w, b, d))
        z_tilde = self.decoder(tgt, c, tgt_mask, memory_key_padding_mask=memory_mask)
        z_tilde = utils.unravel_tensor_batch(z_tilde, self.z_shape)  # HW x B x D --> B x D x H x W

        # new bullshit: maps from B x 1 x D x H x W --> B x L x D x H x W
        z_tilde = z_tilde.sigmoid()
        z_logits = 1 - (self.arange.view(1, -1, 1, 1, 1) - z_tilde.unsqueeze(1)).abs()  # alfredos tricks

        return z_logits

    def quantize(self, c):
        norm = (torch.abs(c.unsqueeze(-1) - self.centroids)) ** 2
        c_tilde = (F.softmax(-1.0 * norm, dim=-1) * self.centroids).sum(-1)
        c_hat = (F.softmax(-1e7 * norm, dim=-1) * self.centroids).sum(-1)
        c_quantized = (c_hat - c_tilde).detach() + c_tilde  # differentiability trick

        return c_quantized

    def mask(self, c, k):
        # c ~ HW x B x D
        # k ~ B
        # todo: Is detach() needed here? k is already detached in sample_k(mu)..

        b, d, h, w = self.z_shape
        mask_from_index = (k * (d * h * w)).to(int).unsqueeze(1)  # B x 1
        mask = ~torch.arange(d * h * w).repeat(b, 1).ge(mask_from_index)  # B x DHW
        mask = mask.view(b, h * w, d).permute(1, 0, 2)  # HW x B x D, reshaped to format of c
        c_masked = c * mask

        # mask for the transformer decoder to know which elements in the sequence that are masked
        memory_mask = torch.arange(h * w).unsqueeze(0).ge(torch.ceil(mask_from_index / float(d)))  # B x HW
        return c_masked, memory_mask

    def _init_parameters(self):
        """
        Initialize parameters in the transformer model.
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def transformer_loss(z, z_tilde_logits, k_transf, log_pk_transf, z_centroids):
    """
    z               ~ B x D x H x W, quantized with z_centroids and masked
    z_tilde_logits  ~ B x L x D x H x W, L is the number of centroids
    k_transf        ~ B
    log_pk_transf   ~ B
    z_centroids     ~ L, symbol centroids for z
    """

    z_mask = ~(z.abs() > 0)

    z_target = (z.unsqueeze(0) - z_centroids.view(-1, 1, 1, 1, 1)).abs().argmin(dim=0)  # map z to centroid index
    z_reconstructed = z_tilde_logits.argmax(dim=1)

    z_target[z_mask] = -1  # set the masked elements to -1
    z_reconstructed[z_mask] = -1  # set the masked elements to -1

    equal = z_target == z_reconstructed
    accuracy = equal.sum(dtype=torch.float) / z.shape.numel()
    perfect_reconstruction = equal.all().item()

    if perfect_reconstruction:
        loss = (k_transf * log_pk_transf).mean()  # policy gradient with reward = -k
    else:
        # ignore_index = -1 sets loss_ijk = 0 for masked elements
        loss = F.cross_entropy(z_tilde_logits, z_target, ignore_index=-1)

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
        # Slice the position embedding with the size of the image
        pos_embedding = self.pos_embedding[:, :, :h, :w]

        return src + pos_embedding


class MuEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model + 1)  # an extra channel for computing mu

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model + 1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2 = self.attention(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # skip connection
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))  # Feedforward
        src = self._pad(src) + self.dropout2(src2)  # skip connection

        # src = self.norm2(src)
        # todo: this layer norm potentially mess up by forcing c values to be small

        out = src[:, :, :-1]  # ~ HW x B x D
        mu = src[:, :, -1].mean(dim=0).sigmoid()  # HW x B take mean over HW --> B

        return out, mu

    def _pad(self, x):
        """
        Pad x with zeros in the channel dimension to allow for skip connection
        """
        hw, b, d = x.shape
        return torch.cat((x, torch.zeros(hw, b, 1)), dim=2)


def main():

    d_ff = 50
    nhead = 2
    d_model = 12  # needs to be same as number of channels of input tensor/image
    dropout = 0.1
    num_centroids = 8
    num_z_centroids = 8
    num_enc_layers = 2
    num_dec_layers = 2

    cifar10_data = CIFAR10(os.getcwd(), train=False, transform=transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(cifar10_data, batch_size=4, shuffle=True)

    transformer = CompressionTransformer(d_model,
                                         nhead,
                                         d_ff,
                                         dropout,
                                         num_centroids,
                                         num_z_centroids,
                                         num_enc_layers,
                                         num_dec_layers)

    for i, (x, y) in enumerate(data_loader):
        if i > 0:
            break

        x = x.repeat(1, 4, 1, 1)  # just quadruple number of channels for testing
        x_flat = utils.flatten_tensor_batch(x)

        c = transformer(x)

        x_hat = utils.unravel_tensor_batch(c, x.shape)
        x_hat_sliced = x_hat[:, :3, :, :]

        print(x)
        print(x.shape)

        print(y)
        print(y.shape)


if __name__ == '__main__':
    main()