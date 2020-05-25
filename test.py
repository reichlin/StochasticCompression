import sys
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import network_test as network
import matplotlib.pyplot as plt
from utils import *


def test_channels_order(model, test_loader):

    model.eval()
    with torch.no_grad():

        for images, _ in test_loader:

            acc_channels, acc_channels_cumulative_z = model.get_information_content(images)

            print()


def print_RD_curve(model, test_loader):

    model.eval()
    with torch.no_grad():

        distortion = []
        compression = []

        for images, _ in test_loader:

            accuracy,  x_hat, mu = model.get_accuracy(images)
            z_size = torch.sum(torch.ceil(mu.detach() * model.n), (1, 2))
            compression.append(bpp(z_size, model.L, mu.shape[1], mu.shape[2], model.n, 512., 768.))

            distortion.append(torch.mean(accuracy).clone().item())

        print()

    return


def main():

    torch.manual_seed(1234)


    ''' PARAMETERS '''
    colors = 3
    model_depth = 3
    model_size = 64
    n = 64
    L = 8
    decoder_type = 0  # 0:deconvolution, 1:upsampling_nearest, 2:upsampling_bilinear

    a_depth = 6
    a_size = 32
    a_act = 1  # 0:relu, 1:leakyrelu

    adaptive_compression_sampling = False

    # POLICY SEARCH
    ''' MODEL DEFINITION '''

    model = network.Net(colors,
                        model_depth,
                        model_size,
                        n,
                        L,
                        adaptive_compression_sampling,
                        a_size,
                        a_depth,
                        a_act,
                        decoder_type)

    ''' DATASET LOADER '''
    trans_test = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.ImageFolder(root="../data/Kodak", transform=trans_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=8)

    device = torch.device('cpu')
    model.load_state_dict(torch.load('../models/experiment_6_epoch_10.pt', map_location=device))
    model.eval()

    frq_symbols = np.zeros([24, 96*64, 64, L+1])
    symbols = torch.linspace((-L / 2.) * model_size, (L / 2.) * model_size, L)

    mus = None

    for b, (images, _) in tqdm(enumerate(test_loader)):

        _, z, mu, _ = model.E(images)
        z, _ = model.quantize(z)
        z = model.mask_z(z, mu)

        if mus is None:
            mus = torch.flatten(mu).detach().cpu()
        else:
            np.concatenate()

        # for h in range(z.shape[2]):
        #     for w in range(z.shape[3]):
        #         for i in range(n):
        #             symbol = z[0, i, h, w].item()
        #             if symbol == 0:
        #                 c = 0
        #             else:
        #                 c = torch.argmin(torch.abs(symbol-symbols)).item() + 1
        #
        #             frq_symbols[b, h*z.shape[3]+w, i, c] += 1

    print()


        # for j in range(1, 11):
        #
        #
        #
        #     _, z, mu, _ = model.E(images)
        #
        #     for k in range(65):
        #         frqs[j-1, k] = torch.sum((torch.ceil(mu * 64) == k) * 1.0) / (mu.shape[1] * mu.shape[2])
        #
        #     bpp = 0
        #     for k in range(0, 65):
        #         bpp += (frqs[j-1, k]*96.*64.*k*np.log2(L))/(512.*768.)
        #     print(bpp)
        #     print()
        #
        #     #mu_n_ceil = torch.flatten(torch.ceil(mu * 64)).detach().numpy()
        #     #plt.hist(mu_n_ceil, bins)
        # #plt.show()
        #
        # break


    # for i, (images, _) in enumerate(test_loader):
    #     x_hat, x_hat_m, mu, z0, z = model(images)
    #     loss_d, accuracy, accuracy_mean, accuracy_m_mean, img_err = model.get_loss_d(images, x_hat, x_hat_m)
    #     model.print_tensor4d(x_hat_m)
    #     print(i, accuracy_m_mean)

    test_channels_order(model, test_loader)
    print_RD_curve(model, test_loader)



if __name__ == '__main__':
    main()
