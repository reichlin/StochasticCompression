import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import MS_SSIM
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from torch.distributions.categorical import Categorical
from torch.distributions.exponential import Exponential


class MS_SSIM(MS_SSIM):
    def forward(self, img1, img2):
        return super(MS_SSIM, self).forward(img1, img2)


class ResidualMiniBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualMiniBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=(1, 1))
        self.BN1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=(1, 1))
        self.BN2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        z = self.conv1(x)
        z = self.BN1(z)
        z = F.relu(z)
        z = self.conv2(z)
        z = self.BN2(z)

        y = x + z

        return y


class ResidualMaxiBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualMaxiBlock, self).__init__()

        self.block1 = ResidualMiniBlock(channels)
        self.block2 = ResidualMiniBlock(channels)
        self.block3 = ResidualMiniBlock(channels)

    def forward(self, x):
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)

        y = x + z

        return y


class BodyModel(nn.Module):

    def __init__(self, channels, depth):
        super(BodyModel, self).__init__()
        self.depth = depth

        self.blocks_E = nn.ModuleList([ResidualMaxiBlock(channels) for _ in range(depth)])
        self.blocks_E.append(ResidualMiniBlock(channels))

    def forward(self, x):
        tmp = x
        for i, conv in enumerate(self.blocks_E):
            tmp = conv(tmp)
        y = x + tmp

        return y


class Net(nn.Module):

    def __init__(self, colors, depth, model_size, n, L, adaptive_compression_sampling, a_size, a_depth, a_act, decoder_type):
        super(Net, self).__init__()

        # constants for the architecture of the model
        self.n = n
        self.depth = depth
        self.colors = colors

        self.adaptive_compression_sampling = adaptive_compression_sampling

        self.a_size = a_size
        self.a_depth = a_depth
        self.a_act = a_act

        # constants for the quantization step
        self.L = L
        self.c = torch.nn.Parameter(torch.linspace((-self.L / 2.) * model_size, (self.L / 2.) * model_size, self.L), requires_grad=True)

        # constants for the random variable mu
        self.probs = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.mask_linspace = torch.nn.Parameter(torch.linspace(0., float(n - 1), n), requires_grad=False)
        self.kernel = torch.nn.Parameter(torch.ones((1, colors, 8, 8)), requires_grad=False)

        # autoencoder
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=colors, out_channels=int(model_size / 2), kernel_size=5, stride=2, padding=(0, 0)),
            nn.BatchNorm2d(int(model_size / 2)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=int(model_size / 2), out_channels=model_size, kernel_size=5, stride=2, padding=(0, 0)),
            nn.BatchNorm2d(model_size),
            nn.ReLU(),
            BodyModel(model_size, depth),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=model_size, out_channels=(self.n + 1), kernel_size=5, stride=2, padding=(0, 0)))

        if decoder_type == 1:
            up_mode = 'nearest'
        else:
            up_mode = 'bilinear'

        if decoder_type == 0:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.n, out_channels=model_size, kernel_size=6, stride=2, padding=(2, 2)),
                nn.BatchNorm2d(model_size),
                nn.ReLU(),
                BodyModel(model_size, depth),
                nn.ConvTranspose2d(in_channels=model_size, out_channels=int(model_size / 2), kernel_size=6, stride=2, padding=(2, 2)),
                nn.BatchNorm2d(int(model_size / 2)),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=int(model_size / 2), out_channels=colors, kernel_size=6, stride=2, padding=(2, 2)))
        else:
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=up_mode),
                nn.Conv2d(in_channels=self.n, out_channels=model_size, kernel_size=5, stride=1, padding=(2, 2)),
                nn.BatchNorm2d(model_size),
                nn.ReLU(),
                BodyModel(model_size, depth),
                nn.Upsample(scale_factor=2, mode=up_mode),
                nn.Conv2d(in_channels=model_size, out_channels=int(model_size / 2), kernel_size=5, stride=1, padding=(2, 2)),
                nn.BatchNorm2d(int(model_size / 2)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode=up_mode),
                nn.Conv2d(in_channels=int(model_size / 2), out_channels=colors, kernel_size=5, stride=1, padding=(2, 2)))

        if self.a_act == 0:
            a_act = nn.ReLU()
        else:
            a_act = nn.LeakyReLU()

        if self.a_depth == 6:
            self.a_1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=self.n, kernel_size=3, stride=1, padding=(1, 1)))

            self.a_2 = nn.Sequential(
                nn.Conv2d(in_channels=(2 * self.n), out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=int(self.a_size / 2.), kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(int(self.a_size / 2.)),
                a_act)

        elif self.a_depth == 8:
            self.a_1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=self.n, kernel_size=3, stride=1, padding=(1, 1)))

            self.a_2 = nn.Sequential(
                nn.Conv2d(in_channels=(2 * self.n), out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=self.a_size, kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(self.a_size),
                a_act,
                nn.Conv2d(in_channels=self.a_size, out_channels=int(self.a_size / 2.), kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(int(self.a_size / 2.)),
                a_act)

        if self.adaptive_compression_sampling:
            self.Alpha = nn.Conv2d(in_channels=int(self.a_size / 2.), out_channels=4, kernel_size=3, stride=1, padding=(1, 1))
        else:
            self.Alpha = nn.Conv2d(in_channels=int(self.a_size / 2.), out_channels=1, kernel_size=3, stride=1, padding=(1, 1))

        self.msssim_k = MS_SSIM(data_range=1.0, size_average=False, channel=3)
        self.msssim_c = MS_SSIM(data_range=1.0, size_average=False, channel=3)
        self.msssim_inf = MS_SSIM(data_range=1.0, size_average=False, channel=3)
        self.msssim_test = MS_SSIM(data_range=255, size_average=False, channel=3)

    def E(self, x):

        z = self.encoder(x)

        z0 = z[:, 0:1, :, :]  # [B, H, W]
        z = z[:, 1:, :, :]  # [B, n, H, W]

        a = self.a_1(z0)
        a = torch.cat((a, z.detach()), 1)
        a = self.a_2(a)  # [B, 1, H, W]
        a = self.Alpha(a)  # [B, {1 or 4}, H, W] 1 if non adaptive, 4 if adaptive

        if self.adaptive_compression_sampling:

            alfa = torch.sigmoid(a)
            mu = alfa[:, 0, :, :]

            alpha = alfa[:, 1, :, :] * (2. - 0.01) + 0.01
            l = alfa[:, 2, :, :] * (float(self.n / 2.) - 1.) + 1.
            h = alfa[:, 3, :, :] * (float(self.n) - (l + 3)) + (l + 3)

            # self.dummy_l = torch.zeros(l.shape, requires_grad=True)
            # l = l + self.dummy_l

        else:
            mu = torch.sigmoid(a[:, 0, :, :])
            alpha = None
            l = None
            h = None

        return z, mu, alpha, l, h, a

    def D(self, z):

        x = self.decoder(z)
        x_hat = torch.sigmoid(x)

        return x_hat

    def sample_k(self, mu):

        n = float(self.n)

        m = Poisson(mu * n)
        k = torch.clamp(m.sample(), 0., n)

        log_pk = m.log_prob(k)
        k = (k / n).detach()

        return k, log_pk

    def sample_bounded_Pareto(self, size, a, l, h):
        u = torch.FloatTensor(size).uniform_()
        num = (u * torch.pow(h, a) - u * torch.pow(l, a) - torch.pow(h, a))
        den = (torch.pow(h, a) * torch.pow(l, a))
        x = torch.pow(- num / den, (-1. / a))
        return x

    def bounded_Pareto_prob(self, x, a, l, h):

        # self.dummy_lp = torch.zeros(l.shape, requires_grad=True)
        # l = l + self.dummy_lp

        num = a * torch.pow(l, a) * torch.pow(x, (-a - 1))
        den = 1 - torch.pow((l / h), a)
        p = num / den

        return p

    def sample_compression(self, mu, alpha, l, h):

        if self.adaptive_compression_sampling:

            # self.dummy_lprima = torch.zeros(l.shape, requires_grad=True)
            # l = l + self.dummy_lprima

            k_compression = (torch.clamp(self.sample_bounded_Pareto(mu.shape, alpha, l, h), 0, self.n)).detach()
            # log_p_compression = torch.log(self.bounded_Pareto_prob(k_compression, alpha, l, h))

            p = self.bounded_Pareto_prob(k_compression, alpha, l, h)
            # self.dummy_p = torch.zeros(p.shape, requires_grad=True)
            # p = p + self.dummy_p

            log_p_compression = torch.log(p)

            # self.dummy_log_p = torch.zeros(log_p_compression.shape, requires_grad=True)
            # log_p_compression = log_p_compression + self.dummy_log_p

            k_compression = k_compression / self.n

        else:

            log_p_compression = None

            l = torch.ceil(mu - mu / 2.)
            h = torch.clamp(torch.ceil(mu + mu / 2.), 3.0)
            alpha = 1.16
            k_compression = torch.clamp(self.sample_bounded_Pareto(mu.shape, alpha, l, h), 0, self.n) / self.n

        return k_compression, log_p_compression

    def quantize(self, z):

        norm = (torch.abs(z.unsqueeze(-1) - self.c)) ** 2
        z_tilde = (F.softmax(-1.0 * norm, dim=-1) * self.c).sum(-1)
        z_hat = (F.softmax(-1e7 * norm, dim=-1) * self.c).sum(-1)
        z_bar = (z_hat - z_tilde).detach() + z_tilde

        quantization_error = torch.mean(torch.abs(z_tilde - z_hat))

        return z_bar, quantization_error

    ''' k [0, 1] '''

    def mask_z(self, z, k):

        mask_linspace = self.mask_linspace.view(1, -1, 1, 1).repeat(z.shape[0], 1, z.shape[2], z.shape[3])
        mask = (~mask_linspace.ge(k.unsqueeze(1) * float(self.n))).detach()
        zm = z * mask

        return zm

    def forward(self, x):

        z, mu, alpha, l, h, a = self.E(x)

        k, log_pk = self.sample_k(mu)
        k_compression, log_p_compression = self.sample_compression(mu, alpha, l, h)

        z, quantization_error = self.quantize(z)

        z_k = self.mask_z(z, k)
        z_compress = self.mask_z(z, k_compression)

        x_hat_k = self.D(z_k)
        x_hat_compress = self.D(z_compress)

        return x_hat_compress, x_hat_k, k, k_compression, mu, a, z, quantization_error, log_pk, log_p_compression, l, h, alpha

    def get_loss_d(self, x, x_hat_compress, x_hat_k):

        img_err_k = torch.abs(x - x_hat_k)
        img_err_c = torch.abs(x - x_hat_compress)

        msssim_compress = self.msssim_c(x, x_hat_compress)
        msssim_k = self.msssim_k(x, x_hat_k)
        msssim_mean_compress = torch.mean(100 * msssim_compress)
        msssim_mean_k = torch.mean(100 * msssim_k)

        loss_distortion = 1 - msssim_compress
        accuracy_k = 100 * msssim_k
        accuracy_c = 100 * msssim_compress

        accuracy_compression_mean = msssim_mean_compress
        accuracy_k_mean = msssim_mean_k

        loss = torch.mean(loss_distortion)

        return loss, accuracy_k, accuracy_c, accuracy_compression_mean, accuracy_k_mean, img_err_k, img_err_c

    def get_loss_k(self, x, img_err_k, img_err_c, accuracy_k, accuracy_c, k, k_compression, log_pk, log_p_compression, a, L2_a):

        kernel = self.kernel
        R_k = F.conv2d(img_err_k, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R_k = torch.squeeze(1. - R_k / (self.colors * kernel.shape[-2] * kernel.shape[-1]))
        R_c = F.conv2d(img_err_c, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R_c = torch.squeeze(1. - R_c / (self.colors * kernel.shape[-2] * kernel.shape[-1]))

        cond_k = accuracy_k.le(97.0)
        avg_cond = torch.mean(cond_k.float())
        cond_k = torch.unsqueeze(torch.unsqueeze(cond_k, -1), -1).repeat(1, R_k.shape[0], R_k.shape[1])
        cond_c = accuracy_c.le(97.0)
        cond_c = torch.unsqueeze(torch.unsqueeze(cond_c, -1), -1).repeat(1, R_c.shape[0], R_c.shape[1])

        R_k = (cond_k * R_k + ~cond_k * -k).detach()
        R_compression = (cond_c * R_c + ~cond_c * -k_compression).detach()

        loss_a = L2_a * torch.sum(torch.pow(a, 2), (1, 2))

        loss_pk = torch.sum(- R_k * log_pk, (1, 2))
        loss_p_compression = torch.sum(- R_compression * log_p_compression, (1, 2))
        loss = torch.mean(loss_pk + loss_p_compression + loss_a)

        return loss, avg_cond, R_k, R_compression

    def get_accuracy(self, x):

        _, x_hat, _, _, mu, _, z, _ = self.forward(x)

        x_uint8 = ((x * 255).type(torch.IntTensor)).type(torch.FloatTensor)
        x_hat_uint8 = (x_hat * 255)

        accuracy = 100 * self.msssim_test(x_uint8, x_hat_uint8)

        return accuracy, x_hat, mu

    def get_information_content(self, x):

        z, mu, _, _, _, _ = self.E(x)
        z, _ = self.quantize(z)

        acc_channels = np.zeros(self.n)

        for i in range(self.n):
            z_m = z * 1.0
            z_m[:, i, :, :] *= 0.0
            x_hat = self.D(z_m)
            acc_channels[i] = torch.mean(100 * self.msssim_inf(x, x_hat))

        acc_channels_cumulative_z = np.zeros(self.n)

        for i in range(self.n):
            z_m = z * 1.0
            z_m[:, i:, :, :] *= 0.0
            x_hat = self.D(z_m)
            acc_channels_cumulative_z[i] = torch.mean(100 * self.msssim_inf(x, x_hat))

        return acc_channels, acc_channels_cumulative_z


