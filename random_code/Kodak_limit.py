import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import MS_SSIM
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
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

    def __init__(self, min_accuracy, model_depth, model_size, n, L, exploration_epsilon):
        super(Net, self).__init__()

        # constants for the architecture of the model
        self.n = n
        self.depth = model_depth
        self.min_accuracy = min_accuracy

        self.exploration_epsilon = exploration_epsilon

        # constants for the quantization step
        self.L = L
        self.c = torch.nn.Parameter(torch.linspace((-self.L/2.)*model_size, (self.L/2.)*model_size, self.L), requires_grad=True)

        self.probs = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.mask_linspace = torch.nn.Parameter(torch.linspace(0., float(n - 1), n), requires_grad=False)
        self.kernel = torch.nn.Parameter(torch.ones((1, 3, 8, 8)), requires_grad=False)

        # autoencoder
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=int(model_size / 2), kernel_size=5, stride=2, padding=(0, 0)),
            nn.BatchNorm2d(int(model_size / 2)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=int(model_size / 2), out_channels=model_size, kernel_size=5, stride=2, padding=(0, 0)),
            nn.BatchNorm2d(model_size),
            nn.ReLU(),
            BodyModel(model_size, model_depth),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=model_size, out_channels=(self.n+1), kernel_size=5, stride=2, padding=(0, 0)))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n, out_channels=model_size, kernel_size=6, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(model_size),
            nn.ReLU(),
            BodyModel(model_size, model_depth),
            nn.ConvTranspose2d(in_channels=model_size, out_channels=int(model_size / 2), kernel_size=6, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(int(model_size / 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=int(model_size / 2), out_channels=3, kernel_size=6, stride=2, padding=(2, 2)))

        self.a_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa1 = nn.BatchNorm2d(32)
        self.a_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa2 = nn.BatchNorm2d(32)
        self.a_conv3 = nn.Conv2d(in_channels=32, out_channels=self.n, kernel_size=3, stride=1, padding=(1, 1))
        self.a_conv4 = nn.Conv2d(in_channels=(2 * self.n), out_channels=32, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa3 = nn.BatchNorm2d(32)
        self.a_conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=(1, 1))
        self.BNa4 = nn.BatchNorm2d(16)
        self.a_conv6 = nn.Conv2d(in_channels=16, out_channels=self.n, kernel_size=3, stride=1, padding=(1, 1))

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
        # a = a + z.detach()
        a = torch.cat((a, z.detach()), 1)
        a = F.relu(self.BNa3(self.a_conv4(a)))
        a = F.relu(self.BNa4(self.a_conv5(a)))
        a = self.a_conv6(a)
        mu = F.softmax(a, dim=1)

        return z0, z, mu, a

    def D(self, z):

        x = self.decoder(z)

        x_hat = torch.sigmoid(x)

        return x_hat

    def sample_k(self, mu):

        mu = torch.transpose(mu, 3, 1)
        m = Categorical(mu.reshape(mu.shape[0] * mu.shape[1] * mu.shape[2], mu.shape[3]))
        k = m.sample()

        cond = (torch.cuda.FloatTensor(k.shape).uniform_()).ge(self.exploration_epsilon)
        k = cond * k + ~cond * (torch.floor(torch.cuda.FloatTensor(k.shape).uniform_() * float(self.n)))

        log_pk = m.log_prob(k)
        k = k.reshape(mu.shape[0], mu.shape[1], mu.shape[2])
        log_pk = log_pk.reshape(mu.shape[0], mu.shape[1], mu.shape[2])

        k = k.type(torch.cuda.FloatTensor) / float(self.n)

        return k, log_pk

    def quantize(self, z):

        norm = (torch.abs(z.unsqueeze(-1) - self.c)) ** 2
        z_tilde = (F.softmax(-1.0 * norm, dim=-1) * self.c).sum(-1)
        z_hat = (F.softmax(-1e7 * norm, dim=-1) * self.c).sum(-1)
        z_bar = (z_hat - z_tilde).detach() + z_tilde

        return z_bar

    def mask_z(self, z, k):

        # mask_linspace = [0, 1, ..., n-1] for every pixel in z
        mask_linspace = self.mask_linspace.view(1, -1, 1, 1).repeat(z.shape[0], 1, z.shape[2], z.shape[3])
        # mask = [1, 1, 1, ..., 1, 0, ..., 0] for every pixel in z, the last value equal to 1 is in the k-th position
        mask = (~mask_linspace.ge(k.unsqueeze(1)*float(self.n))).detach()
        zm = z * mask

        return zm

    def forward(self, x, training=True):

        z0, z, mu, a = self.E(x)

        if training:
            k, log_pk = self.sample_k(mu)  # sample k from mu
        else:
            k = mu.argmax(1)
            k = k.type(torch.cuda.FloatTensor) / float(self.n)
            log_pk = k * 0.0

        z = self.quantize(z)

        z = self.mask_z(z, k)

        x_hat = self.D(z)

        return x_hat, log_pk, k, a, mu, z0, z

    '''
        reconstruction loss, MS-SSIM much better than MSE
    '''
    def get_loss_d(self, x, x_hat):

        # MSE
        mse = torch.mean(torch.pow(x - x_hat, 2.), (1, 2, 3))
        loss_mse = mse
        img_err = torch.abs(x - x_hat)

        # MS-SSIM
        msssim = self.msssim(x, x_hat)
        msssim_accuracy = 100 * msssim
        msssim_mean = torch.mean(msssim_accuracy)
        loss_msssim = 1 - msssim

        if msssim.ge(0.01).all():
            loss_distortion = loss_msssim
        else:
            loss_distortion = loss_mse

        loss = torch.mean(loss_distortion)

        accuracy = msssim_accuracy

        return loss, accuracy, img_err

    def get_loss_k(self, img_err, accuracy, k, log_pk, a, L2_a):

        kernel = self.kernel
        R = F.conv2d(img_err, kernel, bias=None, stride=8, padding=0, dilation=1, groups=1).detach()
        R = torch.squeeze(1. - R/(3*kernel.shape[-2]*kernel.shape[-1]))  # R for each receptive field

        # is the overall error above or below the minimum accuracy?
        cond = accuracy.le(self.min_accuracy)
        avg_cond = torch.mean(cond.float())
        cond = torch.unsqueeze(torch.unsqueeze(cond, -1), -1).repeat(1, R.shape[1], R.shape[2])

        # Reward = accuracy in image reconstruction if accuracy <= self.min_accuracy
        # else inversely proportional to the compression level
        R = (cond * R + ~cond * -k).detach()
        # TODO: we can try a log relationship for the compression reward: -torch.log(1.*self.n*torch.clamp(k, 0.)+1.)

        loss_pk = torch.sum(- R * log_pk, (1, 2))
        loss = torch.mean(loss_pk)

        return loss

    def get_accuracy(self, x):

        x_hat, _, k, _, _, _, z = self.forward(x, False)

        x_uint8 = ((x * 255).type(torch.cuda.IntTensor)).type(torch.cuda.FloatTensor)
        x_hat_uint8 = (x_hat * 255)

        msssim = 100 * self.msssim_test(x_uint8, x_hat_uint8)

        accuracy = msssim

        return accuracy, k


def train(model, train_loader, test_loader, optimizer_d, L2_a, beta, idx_t, writer, device):

    for (images, _) in train_loader:#tqdm(train_loader):

        model.train()

        gpu_imgs = images.to(device).detach()

        x_hat, log_pk, k, a, mu, z0, z = model(gpu_imgs)  # forward pass
        loss_d, accuracy, img_err = model.get_loss_d(gpu_imgs, x_hat)  # compute reconstruction loss
        loss_k = model.get_loss_k(img_err, accuracy, k, log_pk, a, L2_a)

        loss = loss_d + beta * loss_k

        optimizer_d.zero_grad()
        loss.backward(retain_graph=False)
        optimizer_d.step()

        if idx_t % 10 == 0 and idx_t != 0:

            evaluate(model, test_loader, int(idx_t / 10), writer, device)  # test KODAK dataset

        idx_t += 1

    return idx_t


def evaluate(model, test_loader, idx, writer, device):

    model.eval()
    with torch.no_grad():

        distortion = 0
        compression = 0

        for images, _ in test_loader:

            accuracy, k = model.get_accuracy(images.to(device))

            z_size = torch.sum(k.detach() * model.n, (1, 2)).cpu()
            compression += (torch.mean(z_size).item() * np.log2(model.L) + k.shape[1] * k.shape[2] * np.log2(model.n)) / (512. * 768.)

            distortion += torch.mean(accuracy).cpu().clone().numpy()

        writer.add_scalar('test_mean_accuracy', distortion, idx)
        writer.add_scalar('test_mean_compression', compression, idx)


def main():

    torch.manual_seed(1234)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr_d = 0.0003

    exploration_epsilon = 0.5
    exploration_epsilon_decay = 0.05

    min_accuracy = 97.0

    EPOCHS = 10000  # TODO: maybe more?
    model_depth = 3  # TODO: either 3 or 5
    model_size = 64  # TODO: either 64 or 128
    n = 64
    L = 8
    L2_a = 0.0000005
    beta = 0.1

    ''' MODEL DEFINITION '''

    model = Net(min_accuracy, model_depth, model_size, n, L, exploration_epsilon).to(device)

    ''' DATASET LOADER '''
    trans = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.ImageFolder(root="/local_storage/datasets/KODAK_padded", transform=trans)
    train_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=24, shuffle=True, num_workers=8)

    ''' TENSORBOARD WRITER '''

    #/Midgard/home/areichlin/compression
    log_dir = '/Midgard/home/areichlin/compression/log_limit/full_kodak_categorical_decay_'+str(exploration_epsilon_decay)+'_eps0_'+str(exploration_epsilon)
    writer = SummaryWriter(log_dir=log_dir)

    ''' OPTIMIZER, SCHEDULER DEFINITION '''

    optimizer_d = optim.Adam(model.parameters(), lr=lr_d)

    idx_t = 0

    ''' TRAINING LOOP '''

    for epoch in range(1, EPOCHS + 1):
        idx_t = train(model, train_loader, test_loader, optimizer_d, L2_a, beta, idx_t, writer, device)
        if epoch % 1000 == 0 and epoch != 0:
            model.exploration_epsilon -= exploration_epsilon_decay

    writer.close()

    # TODO: add saving parameters of the best model


if __name__ == '__main__':
    main()
