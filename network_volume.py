import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import MS_SSIM
from torch.distributions.bernoulli import Bernoulli


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

    def __init__(self, min_accuracy, colors, depth, model_size, n, L):
        super(Net, self).__init__()

        # constants for the architecture of the model
        self.n = n
        self.min_accuracy = min_accuracy
        self.depth = depth
        self.colors = colors

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

        # model for random variable mu
        # TODO: this is a random architecture, we should find a better one
        self.a_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa1 = nn.BatchNorm2d(32)
        self.a_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa2 = nn.BatchNorm2d(32)
        self.a_conv3 = nn.Conv2d(in_channels=32, out_channels=self.n, kernel_size=3, stride=1, padding=(1, 1))
        self.a_conv4 = nn.Conv2d(in_channels=self.n, out_channels=32, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa3 = nn.BatchNorm2d(32)
        self.a_conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa4 = nn.BatchNorm2d(16)
        self.a_conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=(1, 1))

        # MS-SSIM is used instead of Mean Squared Error for the reconstruction of the original image
        self.msssim = MS_SSIM(data_range=1.0, size_average=False, channel=3, nonnegative_ssim=True)
        self.msssim_test = MS_SSIM(data_range=255, size_average=False, channel=3, nonnegative_ssim=True)

    def E(self, x):

        z = self.encoder(x)

        z0 = z[:, 0:1, :, :]  # [B, H, W], this is a way for the model to understand how useful each part of z is for the reconstruction
        z = z[:, 1:, :, :]  # [B, n, H, W], this is the actual latent representation

        a = F.relu(self.BNa1(self.a_conv1(z0)))
        a = F.relu(self.BNa2(self.a_conv2(a)))
        a = self.a_conv3(a)
        a = a + z
        a = F.relu(self.BNa3(self.a_conv4(a)))
        a = F.relu(self.BNa4(self.a_conv5(a)))
        a = torch.squeeze(self.a_conv6(a))

        mu = torch.sigmoid(a)  # [B, H, W] each value between [0, 1], policy of the agent

        return z, mu, a

    def D(self, z):

        x = self.decoder(z)

        x_hat = torch.sigmoid(x)

        return x_hat

    '''
        given each mu, sample an action and compute the probability of the action given the policy
    '''
    def sample_k(self, mu):

        ''' this first k is a tensor of dimentions [B, H, W] where every value is either -1 or 1'''
        probs = self.probs.view(-1, 1, 1).repeat(mu.shape[0], mu.shape[1], mu.shape[2])
        sampling_distribution = Bernoulli(probs)
        k = sampling_distribution.sample()*2 - 1  # [B, H, W] # {-1, 1}

        ''' this k is either one bit more or one bit less than mu. The idea is to have z a little bit more compressed 
        or less compressed than the current random variable level to see how performances change'''
        n = float(self.n)
        k = ((torch.round(mu * n) + k) / n).detach()

        ''' kind of log probability if the random variable was distributed as a Gaussian'''
        log_pk = - torch.pow((k - mu), 2)

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

    def forward(self, x):

        z, mu, a = self.E(x)  # encoder

        k, log_pk = self.sample_k(mu)  # sample k from mu

        z, quantization_error = self.quantize(z)  # quantize z

        z = self.mask_z(z, k)  # mask z based on sampled k

        x_hat = self.D(z)  # decode masked and quantized z

        return x_hat, log_pk, k, a, mu, z, quantization_error

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
    def get_loss_k(self, img_err, accuracy, k, log_pk, a, L2_a):

        kernel = self.kernel
        R = F.conv2d(img_err, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R = torch.squeeze(1. - R/(self.colors*kernel.shape[-2]*kernel.shape[-1]))  # R for each receptive field

        # is the overall error above or below the minimum accuracy?
        cond = accuracy.le(self.min_accuracy)
        avg_cond = torch.mean(cond.float())
        cond = torch.unsqueeze(torch.unsqueeze(cond, -1), -1).repeat(1, R.shape[1], R.shape[2])

        # Reward = accuracy in image reconstruction if accuracy <= self.min_accuracy
        # else inversely proportional to the compression level
        R = (cond * R + ~cond * -k).detach()
        # TODO: we can try a log relationship for the compression reward: -torch.log(1.*self.n*torch.clamp(k, 0.)+1.)

        loss_pk = torch.sum(- R * log_pk, (1, 2))  # policy gradient
        # TODO: actual reinforcement learning loss can be much more complex, we can make experiments on this

        loss_a = L2_a * torch.sum(torch.pow(a, 2), (1, 2))  # regularization on the guy that computes mu, before the sigmoid
        loss = torch.mean(loss_pk + loss_a)

        return loss, avg_cond, R

    def get_accuracy(self, x):

        x_hat, _, k, a, mu, z_hat, _ = self.forward(x)

        mse = torch.mean(torch.pow(x - x_hat, 2.), (1, 2, 3))
        mse_accuracy = 100. * (1. - mse)

        # in the end the possible values for each pixel has to be within [0, 255]
        x_uint8 = ((x * 255).type(torch.cuda.IntTensor)).type(torch.cuda.FloatTensor)
        x_hat_uint8 = (x_hat * 255)

        msssim = 100 * self.msssim_test(x_uint8, x_hat_uint8)

        accuracy = msssim

        return accuracy, k, x_hat, z_hat, mu, a


