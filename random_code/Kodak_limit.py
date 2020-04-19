import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import MS_SSIM
from tqdm import tqdm
import matplotlib as mpl
from utils import *
mpl.use('Agg')
import matplotlib.pyplot as plt


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

    def __init__(self, depth, model_size, n, L):
        super(Net, self).__init__()

        # constants for the architecture of the model
        self.n = n
        self.depth = depth

        # constants for the quantization step
        self.L = L
        self.c = torch.nn.Parameter(torch.linspace((-self.L/2.)*model_size, (self.L/2.)*model_size, self.L), requires_grad=True)

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
            BodyModel(model_size, depth),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=model_size, out_channels=self.n, kernel_size=5, stride=2, padding=(0, 0)))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n, out_channels=model_size, kernel_size=6, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(model_size),
            nn.ReLU(),
            BodyModel(model_size, depth),
            nn.ConvTranspose2d(in_channels=model_size, out_channels=int(model_size / 2), kernel_size=6, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(int(model_size / 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=int(model_size / 2), out_channels=3, kernel_size=6, stride=2, padding=(2, 2)))

        # MS-SSIM is used instead of Mean Squared Error for the reconstruction of the original image
        self.msssim = MS_SSIM(data_range=1.0, size_average=False, channel=3, nonnegative_ssim=True)
        self.msssim_test = MS_SSIM(data_range=255, size_average=False, channel=3, nonnegative_ssim=True)

    def E(self, x):

        z = self.encoder(x)

        return z

    def D(self, z):

        x = self.decoder(z)

        x_hat = torch.sigmoid(x)

        return x_hat

    def quantize(self, z):

        norm = (torch.abs(z.unsqueeze(-1) - self.c)) ** 2
        z_tilde = (F.softmax(-1.0 * norm, dim=-1) * self.c).sum(-1)
        z_hat = (F.softmax(-1e7 * norm, dim=-1) * self.c).sum(-1)
        z_bar = (z_hat - z_tilde).detach() + z_tilde

        return z_bar

    def forward(self, x):

        z = self.E(x)

        z = self.quantize(z)

        x_hat = self.D(z)

        return x_hat, z

    '''
        reconstruction loss, MS-SSIM much better than MSE
    '''
    def get_loss_d(self, x, x_hat):

        # MSE
        mse = torch.mean(torch.pow(x - x_hat, 2.), (1, 2, 3))
        loss_mse = mse

        # MS-SSIM
        msssim = self.msssim(x, x_hat)
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

        return loss, accuracy, accuracy_mean

    def get_accuracy(self, x):

        x_hat, z = self.forward(x)

        x_uint8 = ((x * 255).type(torch.cuda.IntTensor)).type(torch.cuda.FloatTensor)
        x_hat_uint8 = (x_hat * 255)

        msssim = 100 * self.msssim_test(x_uint8, x_hat_uint8)

        accuracy = msssim

        return accuracy


def train(model, train_loader, test_loader, optimizer_d, idx_t, writer, device):

    for (images, _) in tqdm(train_loader):

        model.train()

        gpu_imgs = images.to(device).detach()

        x_hat, z = model(gpu_imgs)  # forward pass
        loss, accuracy, accuracy_mean = model.get_loss_d(gpu_imgs, x_hat)

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

            accuracy = model.get_accuracy(images.to(device))
            compression += (98. * 98. * model.n * np.log2(model.L) + 98. * 98. * np.log2(model.n)) / (512. * 768.)

            distortion += torch.mean(accuracy).cpu().clone().numpy()

        writer.add_scalar('test_mean_accuracy', distortion, idx)
        writer.add_scalar('test_mean_compression', compression, idx)


def main():

    torch.manual_seed(1234)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr_d = 0.0003

    EPOCHS = 10000  # TODO: maybe more?
    model_depth = 3  # TODO: either 3 or 5
    model_size = 16  # TODO: either 64 or 128
    n = 8
    L = 8

    ''' MODEL DEFINITION '''

    model = Net(model_depth, model_size, n, L).to(device)

    ''' DATASET LOADER '''
    trans = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.ImageFolder(root="/local_storage/datasets/KODAK_padded", transform=trans)
    train_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=24, shuffle=True, num_workers=8)

    ''' TENSORBOARD WRITER '''

    #/Midgard/home/areichlin/compression
    log_dir = './log_limit/n'+str(n)
    writer = SummaryWriter(log_dir=log_dir)

    ''' OPTIMIZER, SCHEDULER DEFINITION '''

    optimizer_d = optim.Adam(model.parameters(), lr=lr_d)

    idx_t = 0
    idx_v = 0

    ''' TRAINING LOOP '''

    for epoch in range(1, EPOCHS + 1):
        idx_t = train(model, train_loader, test_loader, optimizer_d, idx_t, writer, device)

    writer.close()

    # TODO: add saving parameters of the best model


if __name__ == '__main__':
    main()
