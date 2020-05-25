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
                a_act,
                nn.Conv2d(in_channels=int(self.a_size / 2.), out_channels=1, kernel_size=3, stride=1, padding=(1, 1)))

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
                a_act,
                nn.Conv2d(in_channels=int(self.a_size / 2.), out_channels=1, kernel_size=3, stride=1, padding=(1, 1)))

        if self.adaptive_sampling:
            self.Alpha = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=(0, 0)),
                nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=(0, 0)))

        self.msssim = MS_SSIM(data_range=1.0, size_average=False, channel=3)
        self.msssim_test = MS_SSIM(data_range=255, size_average=False, channel=3)

    def E(self, x):

        z = self.encoder(x)

        z0 = z[:, 0:1, :, :]  # [B, H, W]
        z = z[:, 1:, :, :]  # [B, n, H, W]

        a = self.a_1(z0)
        z_in = z.detach()
        a = torch.cat((a, z_in), 1)
        a = self.a_2(a)  # [B, 1, H, W]

        mu = torch.sigmoid(a.view(a.shape[0], a.shape[2], a.shape[3]))

        if self.adaptive_compression_sampling:

            alpha = self.Alpha(a)  # [B, 3, H, W]
            alpha = torch.sigmoid(alpha)

            alpha = alpha[:, 0, :, :] * (2. - 0.01) + 0.01
            l = alpha[:, 1, :, :] * (float(self.n) - 1.) + 1.
            h = alpha[:, 2, :, :] * (float(self.n) - l) + l

        else:

            alpha = None
            l = None
            h = None

        return z, mu, alpha, l, h, a

    def D(self, z):

        x = self.decoder(z)
        x_hat = torch.sigmoid(x)

        return x_hat

    # def sample_bounded_Pareto(self, size, a, l, h):
    #     u = torch.FloatTensor(size).uniform_()
    #     num = (u * torch.pow(h, a) - u * torch.pow(l, a) - torch.pow(h, a))
    #     den = (torch.pow(h, a) * torch.pow(l, a))
    #     x = torch.pow(- num / den, (-1. / a))
    #     return x
    #
    # def bounded_Pareto_prob(self, x, a, l, h):
    #     num = a * torch.pow(l, a) * torch.pow(x, (-a - 1))
    #     den = 1 - torch.pow((l / h), a)
    #     p = num / den
    #     return p
    #
    # def sample_compression(self, mu, alpha, l, h):
    #
    #     if self.adaptive_compression_sampling:
    #
    #         if self.compression_sampling_function == 1:
    #
    #             m = Exponential(alpha)
    #             k_compression = torch.clamp(m.sample() + l, 0, self.n)
    #             log_p_compression = torch.log(alpha * torch.exp(-alpha * k_compression) + l)
    #             k_compression = k_compression / self.n
    #
    #         elif self.compression_sampling_function == 2:
    #
    #             k_compression = torch.clamp(self.sample_bounded_Pareto(mu.shape, alpha, l, h), 0, self.n)
    #             log_p_compression = torch.log(self.bounded_Pareto_prob(k_compression, alpha, l, h))
    #             k_compression = k_compression / self.n
    #
    #     else:
    #
    #         log_p_compression = None
    #
    #         if self.compression_sampling_function == 0:
    #             k_compression_limit = torch.FloatTensor(mu.shape).uniform_() * torch.clamp(mu + 0.02, 0, 1.0)
    #             k_compression = k_compression_limit
    #
    #         elif self.compression_sampling_function == 1:
    #
    #             alpha = -torch.log(0.5) / (mu * self.n)
    #             m = Exponential(alpha)
    #             k_compression = torch.clamp(m.sample() + 1, 0, self.n) / self.n
    #
    #         elif self.compression_sampling_function == 2:
    #             l = torch.ceil(mu - mu / 2.)
    #             h = torch.clamp(torch.ceil(mu + mu / 2.), 3.0)
    #             alpha = 1.16
    #             k_compression = torch.clamp(self.sample_bounded_Pareto(mu.shape, alpha, l, h), 0, self.n) / self.n
    #
    #     return k_compression, log_p_compression

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

        k = mu
        k_compression = mu

        z, quantization_error = self.quantize(z)

        z_k = self.mask_z(z, k)
        z_compress = self.mask_z(z, k_compression)

        x_hat_k = self.D(z_k)
        x_hat_compress = self.D(z_compress)

        return x_hat_compress, x_hat_k, k, k_compression, mu, a, z, quantization_error

    def get_loss_d(self, x, x_hat_compress, x_hat_k):

        img_err = torch.abs(x - x_hat_k)

        msssim_compress = self.msssim(x, x_hat_compress)
        msssim_k = self.msssim(x, x_hat_k)
        msssim_mean_compress = torch.mean(100 * msssim_compress)
        msssim_mean_k = torch.mean(100 * msssim_k)

        loss_distortion = 1 - msssim_compress
        accuracy = 100 * msssim_k

        accuracy_compression_mean = msssim_mean_compress
        accuracy_k_mean = msssim_mean_k

        loss = torch.mean(loss_distortion)

        return loss, accuracy, accuracy_compression_mean, accuracy_k_mean, img_err

    def get_loss_k(self, x, img_err, accuracy, k, k_compression, log_pk, log_p_compression, a, L2_a):

        kernel = self.kernel
        R = F.conv2d(img_err, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R = torch.squeeze(1. - R / (self.colors * kernel.shape[-2] * kernel.shape[-1]))

        cond = accuracy.le(self.min_accuracy)
        avg_cond = torch.mean(cond.float())
        cond = torch.unsqueeze(torch.unsqueeze(cond, -1), -1).repeat(1, R.shape[1], R.shape[2])

        R_k = (cond * R + ~cond * -k).detach()
        R_compression = (cond * R + ~cond * -k_compression).detach()

        loss_a = L2_a * torch.sum(torch.pow(a, 2), (1, 2))

        loss_pk = torch.sum(- R_k * log_pk, (1, 2))
        loss_p_compression = torch.sum(- R_compression * log_p_compression, (1, 2))
        loss = torch.mean(loss_pk + loss_p_compression + loss_a)

        return loss, avg_cond, R

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
            acc_channels[i] = torch.mean(100 * self.msssim(x, x_hat))

        acc_channels_cumulative_z = np.zeros(self.n)

        for i in range(self.n):
            z_m = z * 1.0
            z_m[:, i:, :, :] *= 0.0
            x_hat = self.D(z_m)
            acc_channels_cumulative_z[i] = torch.mean(100 * self.msssim(x, x_hat))

        return acc_channels, acc_channels_cumulative_z


