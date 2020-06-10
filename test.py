import sys
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import network as network
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

    adaptive_compression_sampling = True

    # POLICY SEARCH
    ''' MODEL DEFINITION '''

    model = network.Net(colors,
                        colors,
                        model_depth,
                        model_size,
                        n,
                        L,
                        False,
                        0.15,
                        0.5)

    ''' DATASET LOADER '''
    trans_test = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.ImageFolder(root="./Kodak", transform=trans_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=8)

    device = torch.device('cpu')
    model.load_state_dict(torch.load('./trained_model/Pareto_alpha_0.5_delta_0.15_10.pt', map_location=device), strict=False)
    model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # frq_symbols = np.zeros([24, 96*64, 64, L+1])
    # symbols = torch.linspace((-L / 2.) * model_size, (L / 2.) * model_size, L)

    mus = None

    avg_acc = 0
    avg_c = 0

    for b, (images, _) in tqdm(enumerate(test_loader)):

        accuracy, x_hat, mu = model.get_accuracy(images)

        z_size = torch.sum(mu.detach() * model.n, (1, 2))
        avg_c += bpp(torch.mean(z_size).item(), model.L, mu.shape[1], mu.shape[2], model.n, 512., 768.)
        avg_acc += torch.mean(accuracy).clone().detach().numpy()


        #bpp(torch.mean(torch.sum(k.detach() * model.n, (1, 2))).item(), 8, k.shape[1], k.shape[2], model.n, images.shape[-1], images.shape[-2])

        # x_hat_compress, x_hat_k, k, k_compression, mu, a, z, quantization_error, log_pk, log_p_compression, l, h, alpha = model(images)
        # loss_d, accuracy_k, accuracy_c, accuracy_compression_mean, accuracy_k_mean, img_err_k, img_err_c = model.get_loss_d(images, x_hat_compress, x_hat_k)
        # loss_k, avg_cond, R_k, R_c = model.get_loss_k(images, img_err_k, img_err_c, accuracy_k, accuracy_c, k, k_compression, log_pk, log_p_compression, a, 0.0)
        #
        # loss = loss_d + 0.1 * loss_k
        #
        # optimizer.zero_grad()
        # # model.dummy_l.retain_grad()
        # # model.dummy_lprima.retain_grad()
        # # model.dummy_lp.retain_grad()
        # # model.dummy_p.retain_grad()
        # # model.dummy_log_p.retain_grad()
        # loss.backward(retain_graph=False)
        # # grad_l = model.dummy_l.grad
        # # grad_lprima = model.dummy_lprima.grad
        # # grad_lp = model.dummy_lp.grad
        # # grad_p = model.dummy_p.grad
        # # grad_log_p = model.dummy_log_p.grad
        # # plt.imshow(grad_log_p[0].detach())
        # # plt.show()
        # # dpdl = ((alpha[0, 0, 0] ** 2) * l[0, 0, 0] ** (alpha[0, 0, 0] - 1) * (k_compression[0, 0, 0] * 64) ** (-alpha[0, 0, 0] - 1)) / (
        # #             (1. - (l[0, 0, 0] / h[0, 0, 0]) ** alpha[0, 0, 0]) ** 2)
        # # print(torch.mean((grad > 0)*1.0))
        # optimizer.step()
        #
        # plt.imshow(images.permute(0,2,3,1)[0])
        # plt.show()
        # plt.imshow(x_hat_k.permute(0,2,3,1)[0].detach())
        # plt.show()
        # plt.imshow(x_hat_compress.permute(0, 2, 3, 1)[0].detach())
        # plt.show()
        #
        # print()


        # _, z, mu, _ = model.E(images)
        # z, _ = model.quantize(z)
        # z = model.mask_z(z, mu)
        #
        # if mus is None:
        #     mus = torch.flatten(mu).detach().cpu()
        # else:
        #     np.concatenate()

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

    print("avg compression: " + str(avg_c/24.))
    print("avg accuracy: " + str(avg_acc/24.))


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

    # test_channels_order(model, test_loader)
    # print_RD_curve(model, test_loader)



if __name__ == '__main__':
    main()
