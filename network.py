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

'''
    subpart of both encoder and decoder
'''
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

'''
    subpart of both encoder and decoder
'''
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

'''
    subpart of both encoder and decoder with variable depth and size
'''
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

    def __init__(self, min_accuracy, colors, depth, model_size, n, L, cuda_backend, pareto_interval, pareto_alpha):
        super(Net, self).__init__()

        # constants for the architecture of the model
        self.n = n
        self.min_accuracy = min_accuracy
        self.depth = depth
        self.colors = colors
        self.cuda_backend = cuda_backend

        self.pareto_interval = pareto_interval
        self.pareto_alpha = pareto_alpha

        # constants for the quantization step
        self.L = L
        # c are the centroids for the quantization
        self.c = torch.nn.Parameter(torch.linspace((-self.L/2.)*model_size, (self.L/2.)*model_size, self.L), requires_grad=True)

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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n, out_channels=model_size, kernel_size=6, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(model_size),
            nn.ReLU(),
            BodyModel(model_size, depth),
            nn.ConvTranspose2d(in_channels=model_size, out_channels=int(model_size / 2), kernel_size=6, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(int(model_size / 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=int(model_size / 2), out_channels=colors, kernel_size=6, stride=2, padding=(2, 2)))

        self.a_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.n, kernel_size=3, stride=1, padding=(1, 1)))

        self.a_2 = nn.Sequential(
            nn.Conv2d(in_channels=(2 * self.n), out_channels=32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=(1, 1)))

        self.msssim_k = MS_SSIM(data_range=1.0, size_average=False, channel=3)#, nonnegative_ssim=True)
        self.msssim_c = MS_SSIM(data_range=1.0, size_average=False, channel=3)#, nonnegative_ssim=True)
        self.msssim_inf = MS_SSIM(data_range=1.0, size_average=False, channel=3)#, nonnegative_ssim=True)
        self.msssim_test = MS_SSIM(data_range=255, size_average=False, channel=3)#, nonnegative_ssim=True)

    '''
        Encoder
    '''
    def E(self, x):

        z = self.encoder(x)

        z0 = z[:, 0:1, :, :]  # [B, H, W]
        z = z[:, 1:, :, :]  # [B, n, H, W]

        a = self.a_1(z0)
        a = torch.cat((a, z.detach()), 1)
        a = self.a_2(a)  # [B, 1, H, W]

        mu = torch.sigmoid(a[:, 0, :, :])

        return z, mu

    '''
        Decoder
    '''
    def D(self, z):

        x = self.decoder(z)
        x_hat = torch.sigmoid(x)

        return x_hat

    '''
        given mu in [0, 1] sample k for learning optimal mu and it's log probability ln(p(k | mu))
    '''
    def sample_k(self, mu):

        n = float(self.n)

        m = Poisson(mu * n)
        k = torch.clamp(m.sample(), 0., n)

        log_pk = m.log_prob(k.detach())
        k = (k / n).detach()

        return k, log_pk

    '''
        a = alpha of the bounded pareto -> gives the steepness
        l = lower bound
        h = upper bound
        returns a random sample form this distribution
    '''
    def sample_bounded_Pareto(self, size, a, l, h):
        if self.cuda_backend:
            u = torch.cuda.FloatTensor(size).uniform_()
        else:
            u = torch.FloatTensor(size).uniform_()
        num = (u * torch.pow(h, a) - u * torch.pow(l, a) - torch.pow(h, a))
        den = (torch.pow(h, a) * torch.pow(l, a))
        x = torch.pow(- num / den, (-1. / a))
        return x


    def sample_compression(self, mu):

        delta = mu * self.pareto_interval

        l = torch.clamp(torch.ceil((mu - delta)*self.n), 1.0, float(self.n-3.0))
        h = torch.clamp(torch.ceil((mu + delta)*self.n), 4.0, float(self.n))
        alpha = self.pareto_alpha
        k_compression = torch.clamp(self.sample_bounded_Pareto(mu.shape, alpha, l, h), 0, self.n) / self.n

        return k_compression

    '''
        z_tilde = soft quantization
        z_hat = hard quantization
        z_bar = differentiable quantization
        z_bar will now have one of the L values in self.c
    '''
    def quantize(self, z):

        norm = (torch.abs(z.unsqueeze(-1) - self.c)) ** 2
        z_tilde = (F.softmax(-1.0 * norm, dim=-1) * self.c).sum(-1)
        z_hat = (F.softmax(-1e7 * norm, dim=-1) * self.c).sum(-1)
        z_bar = (z_hat - z_tilde).detach() + z_tilde  # differentiability trick

        return z_bar

    '''
        k [B, H, W] [0, 1]
        mask_linspace = [0, 1, 2, ..., n-1] for each B, H, W
        mask_linspace.ge(k.unsqueeze(1)*float(self.n)) = [0, 0, ..., 0, 1, ..., 1] the first one is in the k-st position
        ~mask_linspace.ge(k.unsqueeze(1)*float(self.n)) = [1, 1, ..., 1, 0, ..., 0]
        for example if k is 0.5 the first channel will be used
                    if k is 1.2 the first two channels will be used
    '''
    def mask_z(self, z, k):

        mask_linspace = self.mask_linspace.view(1, -1, 1, 1).repeat(z.shape[0], 1, z.shape[2], z.shape[3])
        mask = (~mask_linspace.ge(k.unsqueeze(1)*float(self.n))).detach()
        zm = z * mask

        return zm

    def forward(self, x, training=True):

        # encode
        z, mu = self.E(x)

        # compute k for learning optimal mu and k_compress to push information
        if training:
            k, log_pk = self.sample_k(mu)
            k_compression = self.sample_compression(mu)
        else:
            k = mu
            k_compression = mu
            log_pk = None

        # quantize
        z = self.quantize(z)

        # mask z_k and z_c
        z_k = self.mask_z(z, k)
        z_compress = self.mask_z(z, k_compression)

        # decode both
        x_hat_k = self.D(z_k)
        x_hat_compress = self.D(z_compress)

        return x_hat_compress, x_hat_k, log_pk, k, k_compression, mu, z

    '''
        returns the reconstruction loss and img_err to compute the reward
    '''
    def get_loss_d(self, x, x_hat_compress, x_hat_k):

        img_err_k = torch.abs(x - x_hat_k)

        msssim_compress = self.msssim_c(x, x_hat_compress)  # [0, 1] 0: bad, 1: good
        msssim_k = self.msssim_k(x, x_hat_k)
        msssim_mean_compress = torch.mean(100 * msssim_compress)
        msssim_mean_k = torch.mean(100 * msssim_k)

        # if MS-SSIM is 0.0 the derivative is NaN
        if msssim_compress.ge(0.01).all():
            loss_distortion = 1 - msssim_compress
        else:
            loss_distortion = torch.mean(torch.abs(x - x_hat_compress), (1, 2, 3))

        accuracy_k = 100 * msssim_k

        accuracy_compression_mean = msssim_mean_compress
        accuracy_k_mean = msssim_mean_k

        loss = torch.mean(loss_distortion)

        return loss, accuracy_k, accuracy_compression_mean, accuracy_k_mean, img_err_k

    '''
        Reward computation and policy gradient loss
    '''
    def get_loss_k(self, x, img_err_k, accuracy_k, k, log_pk):

        # compute positive reward for the two reconstructed images
        kernel = self.kernel
        R_k = F.conv2d(img_err_k, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R_k = torch.squeeze(1. - R_k/(self.colors*kernel.shape[-2]*kernel.shape[-1]))

        # cond = 0 if above threshold else 1, repeat to same dimentions of mu
        cond_k = accuracy_k.le(self.min_accuracy)
        avg_cond = torch.mean(cond_k.float())
        cond_k = torch.unsqueeze(torch.unsqueeze(cond_k, -1), -1).repeat(1, R_k.shape[1], R_k.shape[2])

        # reward for k, if cond = 1 -> R = accuracy else R = -k
        R_k = (cond_k * R_k + ~cond_k * -k).detach()
        loss_pk = torch.sum(- R_k * log_pk, (1, 2)) # policy gradient

        loss = torch.mean(loss_pk)

        return loss, avg_cond, R_k

    # compute accuracy on testing set, output image not continuous but within [0, 255]
    def get_accuracy(self, x):

        _, x_hat, _, _, _, mu, z = self.forward(x, False)

        if self.cuda_backend:
            x_uint8 = ((x * 255).type(torch.cuda.IntTensor)).type(torch.cuda.FloatTensor)
        else:
            x_uint8 = ((x * 255).type(torch.IntTensor)).type(torch.FloatTensor)
        x_hat_uint8 = (x_hat * 255)

        accuracy = 100 * self.msssim_test(x_uint8, x_hat_uint8)

        return accuracy, x_hat, mu

    '''
        ignore, just for experiments
    '''
    def get_information_content(self, x):

        z, mu = self.E(x)
        z = self.quantize(z)

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


if __name__ == '__main__':
    cuda_backend = torch.cuda.is_available()

    EPOCHS = 10
    Batch_size = 32
    min_accuracy = 97.0
    colors = 3
    model_depth = 3
    model_size = 64
    n = 64
    HW = 168
    L = 8
    L2 = 0.0
    L2_a = 0.0
    decoder_type = 0  # 0:deconvolution, 1:upsampling_nearest, 2:upsampling_bilinear

    lr_d = 0.0003
    gamma = 1.0
    lr_step_size = 1
    clip_gradient = False

    lr_k = 0.0003
    second_optimizer = False

    # POLICY NETWORK
    a_depth = 6
    a_size = 32
    a_act = 1  # 0:relu, 1:leakyrelu

    beta = 0.1
    distortion_training_epochs = 1  # int(sys.argv[2])  # {1, 2, 5}

    # POLICY SEARCH

    k_sampling_policy = 1  # int(sys.argv[3])  # 0:binary, 1:poisson
    exploration_epsilon = 0.0

    compression_sampling_function = 2  # int(sys.argv[4])  # 0:U*mu+0.2, 1:Exponential, 2:Pareto Bounded
    adaptive_compression_sampling = True  # int(sys.argv[5]) == 1  # {False, True}

    pareto_alpha = 1.16  # float(sys.argv[2])
    pareto_interval = 0.5  # float(sys.argv[3])

    model = Net(min_accuracy,
                colors,
                model_depth,
                model_size,
                n,
                L,
                compression_sampling_function,
                adaptive_compression_sampling,
                k_sampling_policy,
                exploration_epsilon,
                a_size,
                a_depth,
                a_act,
                decoder_type,
                cuda_backend,
                pareto_interval,
                pareto_alpha)

    b = 32
    c = 3
    h_tr = w_tr = 168  # msssim fucks up if < 168. If msssim = 0, then d/dx msssim = nan.
    h_te = 768
    w_te = 512

    x_tr = torch.rand(b, c, h_tr, w_tr)
    x_te = torch.rand(b, c, h_te, w_te)

    model(x_tr)