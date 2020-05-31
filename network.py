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

'''
    general network class:
    min_accuracy = accuracy threshold from focusing to accuracy to focusing on compression
    colors = 3
    depth = depth of the body for both encoder and decoder
    model_size = size of the body for both encoder and decoder
    n = length of each z_i patch
    L = number of symbols used in the quantization
    compression_sampling_function = 0: uniform mu, 1: exponential, 2: bounded pareto
    adaptive_compression_sampling = 0: some function of mu, 1: learned parameters
    k_sampling_policy = 0: mu +- 1, 1: poisson
    exploration_epsilon = noise on k sampling
    a_size = size of the mu network
    a_depth = depth of the mu network
    a_act = leaky_relu
    decoder_type = 0: deconvolutions 1: upsampling nearest, 2: upsampling bilinear
    cuda_backend = True: if using a GPU, False: if using a CPU
'''
class Net(nn.Module):

    def __init__(self, min_accuracy, colors, depth, model_size, n, L, compression_sampling_function, adaptive_compression_sampling, k_sampling_policy, exploration_epsilon, a_size, a_depth, a_act, decoder_type, cuda_backend, pareto_interval, pareto_alpha):
        super(Net, self).__init__()

        # constants for the architecture of the model
        self.n = n
        self.min_accuracy = min_accuracy
        self.depth = depth
        self.colors = colors
        self.cuda_backend = cuda_backend

        self.compression_sampling_function = compression_sampling_function  # 0: uniform mu, 1: exponential, 2: bounded pareto
        self.adaptive_compression_sampling = adaptive_compression_sampling
        self.k_sampling_policy = k_sampling_policy  # 0: binary, 1: Poisson
        self.exploration_epsilon = exploration_epsilon

        self.a_size = a_size
        self.a_depth = a_depth
        self.a_act = a_act

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
                nn.Conv2d(in_channels=self.a_size, out_channels=int(self.a_size/2.), kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(int(self.a_size/2.)),
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

        self.msssim_k = MS_SSIM(data_range=1.0, size_average=False, channel=3, nonnegative_ssim=True)
        self.msssim_c = MS_SSIM(data_range=1.0, size_average=False, channel=3, nonnegative_ssim=True)
        self.msssim_inf = MS_SSIM(data_range=1.0, size_average=False, channel=3, nonnegative_ssim=True)
        self.msssim_test = MS_SSIM(data_range=255, size_average=False, channel=3, nonnegative_ssim=True)

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
        a = self.Alpha(a)  # [B, {1 or 4}, H, W] 1 if non adaptive, 4 if adaptive

        if self.adaptive_compression_sampling:

            alfa = torch.sigmoid(a)
            mu = alfa[:, 0, :, :]

            # bound the parameters for the compression sampling distribution
            if self.compression_sampling_function == 0:  # linear sampling
                alpha = None
                l = alfa[:, 1, :, :] * (float(self.n / 2.) - 1.) + 1.
                h = alfa[:, 2, :, :] * (float(self.n) - (l + 3)) + (l + 3)
            elif self.compression_sampling_function == 1:  # exponential sampling
                alpha = alfa[:, 1, :, :] * (2. - 0.01) + 0.01
                l = alfa[:, 2, :, :] * (float(self.n) - 1.) + 1.
                h = None
            elif self.compression_sampling_function == 2:  # bounded pareto sampling
                alpha = alfa[:, 1, :, :] * (2. - 0.01) + 0.01
                l = alfa[:, 2, :, :] * (float(self.n / 2.) - 1.) + 1.
                h = alfa[:, 3, :, :] * (float(self.n) - (l + 3)) + (l + 3)

        else:
            mu = torch.sigmoid(a[:, 0, :, :])
            alpha = None
            l = None
            h = None

        return z, mu, alpha, l, h, a

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

        if self.k_sampling_policy == 0:  # mu +- 1

            probs = self.probs.view(-1, 1, 1).repeat(mu.shape[0], mu.shape[1], mu.shape[2])
            sampling_distribution = Bernoulli(probs)
            k = sampling_distribution.sample() * 2 - 1  # [B, 1, H, W] # {-1, 1}
            if self.cuda_backend:
                k = k * torch.ceil(torch.cuda.FloatTensor(k.shape).uniform_())
            else:
                k = k * torch.ceil(torch.FloatTensor(k.shape).uniform_())

            n = float(self.n)
            k = (torch.round(mu * n) + k) / n
            if self.cuda_backend:
                cond = (torch.cuda.FloatTensor(k.shape).uniform_()).ge(self.exploration_epsilon)
                k = cond * k + ~cond * (torch.round(torch.cuda.FloatTensor(k.shape).uniform_() * n) / n)
            else:
                cond = (torch.FloatTensor(k.shape).uniform_()).ge(self.exploration_epsilon)
                k = cond * k + ~cond * (torch.round(torch.FloatTensor(k.shape).uniform_() * n) / n)

            k = k.detach()

            log_pk = - torch.pow((k.detach() - mu), 2)

        elif self.k_sampling_policy == 1:  # poisson

            n = float(self.n)

            m = Poisson(mu * n)
            k = torch.clamp(m.sample(), 0., n)

            if self.cuda_backend:
                cond = (torch.cuda.FloatTensor(k.shape).uniform_()).ge(self.exploration_epsilon)
                k = cond * k + ~cond * (torch.round(torch.cuda.FloatTensor(k.shape).uniform_() * n)) # [0, n]
            else:
                cond = (torch.FloatTensor(k.shape).uniform_()).ge(self.exploration_epsilon)
                k = cond * k + ~cond * (torch.round(torch.FloatTensor(k.shape).uniform_() * n))  # [0, n]

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

    '''
        returns the log probability of a sample
    '''
    def bounded_Pareto_prob(self, x, a, l, h):
        num = a * torch.pow(l, a) * torch.pow(x, (-a-1))
        den = 1 - torch.pow((l/h), a)
        p = num/den
        return p

    '''
        mu    [B, H, W] [0, 1]
        alpha [B, H, W] [0.01, 2.0]
        l     [B, H, W] [0, n]
        h     [B, H, W] [l, n]
        
        k_compression [B, H, W] [0, 1]
    '''
    def sample_compression(self, mu, alpha, l, h):

        if self.adaptive_compression_sampling:

            if self.compression_sampling_function == 0:  # linear

                if self.cuda_backend:
                    u = torch.cuda.FloatTensor(mu.shape).uniform_()
                else:
                    u = torch.FloatTensor(mu.shape).uniform_()
                k_compression = (u * (h - l) + l).detach()
                log_p_compression = torch.log(1./(h-l))
                k_compression = (k_compression / self.n).detach()

            elif self.compression_sampling_function == 1:  # exponential

                m = Exponential(alpha)
                k_compression = torch.clamp(m.sample() + l, 0, self.n)
                log_p_compression = torch.log(alpha * torch.exp(-alpha*k_compression) + l)
                k_compression = (k_compression / self.n).detach()

            elif self.compression_sampling_function == 2:  # bounded pareto

                k_compression = (torch.clamp(self.sample_bounded_Pareto(mu.shape, alpha, l, h), 0, self.n)).detach()
                log_p_compression = torch.log(self.bounded_Pareto_prob(k_compression, alpha, l, h))
                k_compression = (k_compression / self.n).detach()

        else:

            log_p_compression = None

            if self.compression_sampling_function == 0:  # linear

                if self.cuda_backend:
                    k_compression_limit = torch.cuda.FloatTensor(mu.shape).uniform_() * torch.clamp(mu*self.n + 2., 0, self.n)
                else:
                    k_compression_limit = torch.FloatTensor(mu.shape).uniform_() * torch.clamp(mu*self.n + 2., 0, self.n)
                k_compression = k_compression_limit / self.n

            elif self.compression_sampling_function == 1:  # linear

                alpha = -np.log(0.5)/(mu * self.n)
                m = Exponential(alpha)
                k_compression = torch.clamp(m.sample()+1, 0, self.n) / self.n

            elif self.compression_sampling_function == 2:  # linear

                delta = mu * self.pareto_interval

                l = torch.clamp(torch.ceil((mu - delta)*self.n), 1.0, float(self.n-3.0))
                h = torch.clamp(torch.ceil((mu + delta)*self.n), 4.0, float(self.n))
                alpha = self.pareto_alpha #1.16
                k_compression = torch.clamp(self.sample_bounded_Pareto(mu.shape, alpha, l, h), 0, self.n) / self.n

        return k_compression, log_p_compression

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
        z, mu, alpha, l, h, a = self.E(x)

        # compute k for learning optimal mu and k_compress to push information
        if training:
            k, log_pk = self.sample_k(mu)
            k_compression, log_p_compression = self.sample_compression(mu, alpha, l, h)
        else:
            k = mu
            k_compression = mu
            log_pk = None
            log_p_compression = None

        # quantize
        z = self.quantize(z)

        # mask z_k and z_c
        z_k = self.mask_z(z, k)
        z_compress = self.mask_z(z, k_compression)

        # decode both
        x_hat_k = self.D(z_k)
        x_hat_compress = self.D(z_compress)

        return x_hat_compress, x_hat_k, log_pk, log_p_compression, k, k_compression, mu, a, z

    '''
        returns the reconstruction loss and img_err to compute the reward
    '''
    def get_loss_d(self, x, x_hat_compress, x_hat_k):

        img_err_k = torch.abs(x - x_hat_k)
        img_err_c = torch.abs(x - x_hat_compress)

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
        accuracy_c = 100 * msssim_compress

        accuracy_compression_mean = msssim_mean_compress
        accuracy_k_mean = msssim_mean_k

        loss = torch.mean(loss_distortion)

        return loss, accuracy_k, accuracy_c, accuracy_compression_mean, accuracy_k_mean, img_err_k, img_err_c

    '''
        Reward computation and policy gradient loss
    '''
    def get_loss_k(self, x, img_err_k, img_err_c, accuracy_k, accuracy_c, k, k_compression, log_pk, log_p_compression, a, L2_a):

        # compute positive reward for the two reconstructed images
        kernel = self.kernel
        R_k = F.conv2d(img_err_k, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R_k = torch.squeeze(1. - R_k/(self.colors*kernel.shape[-2]*kernel.shape[-1]))
        R_c = F.conv2d(img_err_c, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R_c = torch.squeeze(1. - R_c / (self.colors * kernel.shape[-2] * kernel.shape[-1]))

        # cond = 0 if above threshold else 1, repeat to same dimentions of mu
        cond_k = accuracy_k.le(self.min_accuracy)
        cond_c = accuracy_c.le(self.min_accuracy)
        avg_cond = torch.mean(cond_k.float())
        cond_k = torch.unsqueeze(torch.unsqueeze(cond_k, -1), -1).repeat(1, R_k.shape[1], R_k.shape[2])
        cond_c = torch.unsqueeze(torch.unsqueeze(cond_c, -1), -1).repeat(1, R_k.shape[1], R_k.shape[2])

        # regularization (is set to 0 now)
        loss_a = L2_a * torch.sum(torch.pow(a, 2), (1, 2, 3))

        # reward for k, if cond = 1 -> R = accuracy else R = -k
        R_k = (cond_k * R_k + ~cond_k * -k).detach()
        loss_pk = torch.sum(- R_k * log_pk, (1, 2)) # policy gradient

        # same idea for compression if it is adaptive
        if self.adaptive_compression_sampling:
            R_compression = (cond_c * R_c + ~cond_c * -k_compression).detach()
            loss_p_compression = torch.sum(- R_compression * log_p_compression, (1, 2))
            loss = torch.mean(loss_pk + loss_p_compression + loss_a)
        else:
            loss = torch.mean(loss_pk + loss_a)

        return loss, avg_cond, R_k

    # compute accuracy on testing set, output image not continuous but within [0, 255]
    def get_accuracy(self, x):

        _, x_hat, _, _, _, _, mu, _, z = self.forward(x, False)

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

        z, mu, _, _, _, _ = self.E(x)
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


