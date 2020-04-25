import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import MS_SSIM
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


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

    def __init__(self, min_accuracy, colors, depth, model_size, n, L, advantage, k_sampling_window, exploration_epsilon, sampling_policy, sigma, a_size, a_depth, a_join, a_detach, a_act):
        super(Net, self).__init__()

        # constants for the architecture of the model
        self.n = n
        self.min_accuracy = min_accuracy
        self.depth = depth
        self.colors = colors

        self.advantage = advantage
        self.k_sampling_window = k_sampling_window
        self.exploration_epsilon = exploration_epsilon

        self.sampling_policy = sampling_policy

        self.sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=False)

        self.a_size = a_size
        self.a_depth = a_depth
        self.a_join = a_join
        self.a_detach = a_detach
        self.a_act = a_act

        # constants for the quantization step
        self.L = L
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
                a_act,
                nn.Conv2d(in_channels=int(self.a_size/2.), out_channels=1, kernel_size=3, stride=1, padding=(1, 1)))

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


        if self.advantage:
            self.Expected_R = nn.Sequential(
                nn.Conv2d(in_channels=(self.n + 1), out_channels=int(model_size / 2), kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(int(model_size / 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=int(model_size / 2), out_channels=int(model_size / 2), kernel_size=3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(int(model_size / 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=int(model_size / 2), out_channels=1, kernel_size=3, stride=1, padding=(1, 1)),
                nn.Tanh())


        # MS-SSIM is used instead of Mean Squared Error for the reconstruction of the original image
        self.msssim = MS_SSIM(data_range=1.0, size_average=False, channel=3, nonnegative_ssim=True)
        self.msssim_test = MS_SSIM(data_range=255, size_average=False, channel=3, nonnegative_ssim=True)

    def E(self, x):

        z = self.encoder(x)

        z_tot = z

        z0 = z[:, 0:1, :, :]  # [B, H, W], this is a way for the model to understand how useful each part of z is for the reconstruction
        z = z[:, 1:, :, :]  # [B, n, H, W], this is the actual latent representation

        a = self.a_1(z0)
        if self.a_detach:
            z_in = z.detach()
        else:
            z_in = z
        if self.a_join == 0:
            a = a + z_in
        else:
            a = torch.cat((a, z_in), 1)
        a = self.a_2(a)

        a = torch.squeeze(a)
        mu = torch.sigmoid(a)  # [B, H, W] each value between [0, 1], policy of the agent

        return z_tot, z0, z, mu, a

    def D(self, z):

        x = self.decoder(z)

        x_hat = torch.sigmoid(x)

        return x_hat

    '''
        given each mu, sample an action and compute the probability of the action given the policy
    '''
    def sample_k(self, mu):

        if self.sampling_policy == 0:
            probs = self.probs.view(-1, 1, 1).repeat(mu.shape[0], mu.shape[1], mu.shape[2])
            sampling_distribution = Bernoulli(probs)
            k = sampling_distribution.sample() * 2 - 1  # [B, H, W] # {-1, 1}
            k = k * torch.ceil(torch.cuda.FloatTensor(k.shape).uniform_() * self.k_sampling_window)

            n = float(self.n)
            k = ((torch.round(mu * n) + k) / n).detach()

            cond = (torch.cuda.FloatTensor(k.shape).uniform_()).ge(self.exploration_epsilon)
            k = cond * k + ~cond * (torch.round(torch.cuda.FloatTensor(k.shape).uniform_() * n) / n)

            log_pk = - torch.pow((k - mu), 2)

        elif self.sampling_policy == 1:

            m = Normal(mu, self.sigma.repeat(mu.shape[0], mu.shape[1], mu.shape[2]))
            k = m.sample()

            log_pk = m.log_prob(k)

        # elif self.sampling_policy == 2:
        #
        #     mu = torch.transpose(mu, 3, 1)
        #     m = Categorical(mu.reshape(mu.shape[0] * mu.shape[1] * mu.shape[2], mu.shape[3]))
        #     k = m.sample()
        #
        #     cond = (torch.cuda.FloatTensor(k.shape).uniform_()).ge(self.exploration_epsilon)
        #     k = cond * k + ~cond * (torch.floor(torch.cuda.FloatTensor(k.shape).uniform_() * float(self.n)))
        #
        #
        #     log_pk = m.log_prob(k)
        #     k = k.reshape(mu.shape[0], mu.shape[1], mu.shape[2])
        #     log_pk = log_pk.reshape(mu.shape[0], mu.shape[1], mu.shape[2])
        #
        #     k = k.type(torch.cuda.FloatTensor) / float(self.n)
        #     entropy = None
        #
        # elif self.sampling_policy == 3:
        #
        #     m = Normal(mu, sigma)
        #     k = m.sample()
        #
        #     entropy = m.entropy()
        #
        #     log_pk = m.log_prob(k)

        return k, log_pk

    ''' This function takes z and quantize it so that the only possible values in it are the ones in self.c '''
    def quantize(self, z):

        norm = (torch.abs(z.unsqueeze(-1) - self.c)) ** 2
        z_tilde = (F.softmax(-1.0 * norm, dim=-1) * self.c).sum(-1)  # differentiable quantization
        z_hat = (F.softmax(-1e7 * norm, dim=-1) * self.c).sum(-1)  # hard quantization
        # trick so that in the forward pass the hard quantization is used and in the backward the differentiable quantization is used
        z_bar = (z_hat - z_tilde).detach() + z_tilde

        quantization_error = torch.mean(torch.abs(z_tilde-z_hat))

        return z_bar, quantization_error

    '''
        given z and the sampled k, mask z to remove allegedly useless bits and overcompress the image
    '''
    def mask_z(self, z, k):

        # mask_linspace = [0, 1, ..., n-1] for every pixel in z
        mask_linspace = self.mask_linspace.view(1, -1, 1, 1).repeat(z.shape[0], 1, z.shape[2], z.shape[3])
        # mask = [1, 1, 1, ..., 1, 0, ..., 0] for every pixel in z, the last value equal to 1 is in the k-th position
        mask = (~mask_linspace.ge(k.unsqueeze(1)*float(self.n))).detach()
        zm = z * mask

        return zm

    def forward(self, x, training=True):

        z_tot, z0, z, mu, a = self.E(x)  # encoder

        if training:
            k, log_pk = self.sample_k(mu)  # sample k from mu
        else:
            # if self.sampling_policy == 2:
            #     k = mu.argmax(1)
            #     k = k.type(torch.cuda.FloatTensor) / float(self.n)
            #     log_pk = k * 0.0
            #     entropy = None
            # else:
            k = mu
            log_pk = None

        z, quantization_error = self.quantize(z)  # quantize z

        z = self.mask_z(z, k)  # mask z based on sampled k

        x_hat = self.D(z)  # decode masked and quantized z

        return x_hat, log_pk, k, a, mu, z_tot, z0, z, quantization_error

    '''
        reconstruction loss, MS-SSIM much better than MSE
    '''
    def get_loss_d(self, x, x_hat):

        # MSE
        mse = torch.mean(torch.pow(x - x_hat, 2.), (1, 2, 3))
        loss_mse = mse
        mse_accuracy = 100. * (1. - mse)
        mse_accuracy_mean = torch.mean(mse_accuracy)
        img_err = torch.abs(x - x_hat)

        # MS-SSIM
        msssim = self.msssim(x, x_hat)  # [0, 1] 0: big error, 1: no error
        msssim_accuracy = 100 * msssim
        msssim_mean = torch.mean(msssim_accuracy)
        loss_msssim = 1 - msssim

        if msssim.ge(0.01).all():
            loss_distortion = loss_msssim
        else:
            loss_distortion = loss_mse

        accuracy = msssim_accuracy

        accuracy_mean = msssim_mean

        loss = torch.mean(loss_distortion)

        return loss, accuracy, accuracy_mean, img_err

    '''
        Reinforcement learning loss to learn the optimal mu for each image
    '''
    def get_loss_k(self, img_err, accuracy, k, z_tot, log_pk, a, L2_a):

        kernel = self.kernel
        R = F.conv2d(img_err, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R = torch.squeeze(1. - R/(self.colors*kernel.shape[-2]*kernel.shape[-1]))  # R for each receptive field

        # is the overall error above or below the minimum accuracy?
        cond = accuracy.le(self.min_accuracy)
        #cond = (torch.mean(accuracy)).le(self.min_accuracy)
        avg_cond = torch.mean(cond.float())
        cond = torch.unsqueeze(torch.unsqueeze(cond, -1), -1).repeat(1, R.shape[1], R.shape[2])

        # Reward = accuracy in image reconstruction if accuracy <= self.min_accuracy
        # else inversely proportional to the compression level
        R = (cond * R + ~cond * -k).detach()
        # TODO: we can try a log relationship for the compression reward: -torch.log(1.*self.n*torch.clamp(k, 0.)+1.)

        loss_a = L2_a * torch.sum(torch.pow(a, 2), (1, 2))  # regularization on the guy that computes mu, before the sigmoid

        if self.training and self.advantage:
            E_r = torch.squeeze(self.Expected_R(z_tot.detach()))
            loss_pk = torch.sum(- (R - E_r) * log_pk, (1, 2))
            loss = torch.mean(loss_pk + loss_a) + torch.mean( torch.pow( (E_r - R), 2) )
            adv_err = torch.mean( torch.pow( (E_r - R), 2) )
        else:
            # if self.sampling_policy == 3:
            #     loss = torch.mean(torch.sum(- R * log_pk - self.entropy_bonus * entropy, (1, 2)))
            # else:
            loss_pk = torch.sum(- R * log_pk, (1, 2))
            loss = torch.mean(loss_pk + loss_a)
            adv_err = loss * 0.0

        return loss, avg_cond, R, adv_err

    def get_accuracy(self, x):

        x_hat, _, k, a, mu, _, z0, z_hat, _ = self.forward(x, False)

        mse = torch.mean(torch.pow(x - x_hat, 2.), (1, 2, 3))
        mse_accuracy = 100. * (1. - mse)

        # in the end the possible values for each pixel has to be within [0, 255]
        x_uint8 = ((x * 255).type(torch.cuda.IntTensor)).type(torch.cuda.FloatTensor)
        x_hat_uint8 = (x_hat * 255)

        msssim = 100 * self.msssim_test(x_uint8, x_hat_uint8)

        accuracy = msssim

        return accuracy, k, x_hat, z0, z_hat, mu, a


