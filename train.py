import time
import torch.nn as nn
from torchvision.datasets import ImageNet
from tqdm import tqdm
import network_volume as network
import matplotlib as mpl
from utils import *
mpl.use('Agg')
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, optimizer_d, optimizer_k, second_optimizer, L2_a, beta, idx_t, writer, device):

    for (images, _) in tqdm(train_loader):

        model.train()

        gpu_imgs = images.to(device).detach()

        x_hat, log_pk, k, a, mu, z0, z_hat, quantization_error = model(gpu_imgs)  # forward pass
        loss_d, accuracy, accuracy_mean, img_err = model.get_loss_d(gpu_imgs, x_hat)  # compute reconstruction loss
        loss_k, avg_cond, R = model.get_loss_k(img_err, accuracy, k, log_pk, a, L2_a)  # compute reinforcement learning loss

        ''' bunch of constants for the tensorboard '''
        z_size = torch.sum(k.detach() * model.n, (1, 2)).cpu()
        z_size_mean = torch.mean(z_size).item()
        z_size_max = torch.max(z_size).item()
        z_size_min = torch.min(z_size).item()
        accuracy_mean = accuracy_mean.item()
        avg_cond = avg_cond.item()
        R = torch.mean(R).item()
        mu_img = mu.detach().cpu().clone().numpy()
        mu_img = mu_img / np.expand_dims(np.expand_dims(np.max(mu_img, (1, 2)), -1), -1)
        a = torch.mean(a).item()
        c_span = torch.abs(torch.max(model.c.detach().cpu()) - torch.min(model.c.detach().cpu())).item()
        k_values = torch.flatten(k).detach().cpu()

        # a_prima = model.a_prima.detach().cpu().numpy()
        # a_dopo = model.a_dopo.detach().cpu().numpy()
        # a_weights = model.a_conv6.weight.detach().cpu().numpy()

        # a_prima_neg = np.average(a_prima <= 0.)


        ''' either optimize together or separately '''
        if second_optimizer:
            optimizer_k.zero_grad()
            # model.dummy_a.retain_grad()
            loss_k.backward(retain_graph=True)
            # grad_a = torch.flatten(model.dummy_a.grad)
            optimizer_k.step()

            optimizer_d.zero_grad()
            loss_d.backward(retain_graph=False)
            optimizer_d.step()
        else:
            loss = loss_d + beta * loss_k

            optimizer_d.zero_grad()
            loss.backward(retain_graph=False)
            optimizer_d.step()

        ''' compute the bit per pixels of the images '''
        bpp_mean = bpp(k.shape[1], k.shape[2], model.n, z_size_mean, images.shape[-1]*images.shape[-2], np.log2(model.L))
        bpp_max = bpp(k.shape[1], k.shape[2], model.n, z_size_max, images.shape[-1]*images.shape[-2], np.log2(model.L))
        bpp_min = bpp(k.shape[1], k.shape[2], model.n, z_size_min, images.shape[-1]*images.shape[-2], np.log2(model.L))

        writer.add_scalar('training_accuracy', accuracy_mean, idx_t)
        writer.add_scalar('compression_average', bpp_mean, idx_t)
        writer.add_scalar('compression_max', bpp_max, idx_t)
        writer.add_scalar('compression_min', bpp_min, idx_t)
        writer.add_scalar('mean_quantization_error', quantization_error.item(), idx_t)
        writer.add_scalar('span_centroids', c_span, idx_t)
        writer.add_scalar('cond_average', avg_cond, idx_t)
        writer.add_scalar('R_value_average', R, idx_t)
        writer.add_scalar('mu_value_average', a, idx_t)
        writer.add_histogram('k', k_values, idx_t)
        # writer.add_histogram('grad_a', grad_a, idx_t)
        #
        # writer.add_scalar('ratio_a_prima_neg', a_prima_neg, idx_t)
        # writer.add_histogram('a_dopo', a_dopo, idx_t)
        # writer.add_histogram('a_6_w', a_weights, idx_t)

        if idx_t % 1000 == 0 and idx_t != 0:

            imgs_x = images[0:3].detach().clone().numpy()
            imgs_x_hat = x_hat[0:3].detach().cpu().clone().numpy()
            imgs_mu_img = (np.expand_dims(mu_img, 1))[0:3]
            z_img = z_hat[0:3,0:1].detach().cpu().clone().numpy()
            z_img_sum = np.expand_dims(torch.sum(z_hat, dim=1)[0:3].detach().cpu().clone().numpy(), 1)
            z0_img = z0[0:3].detach().cpu().clone().numpy()
            writer.add_images("x_hat", imgs_x_hat, int(idx_t / 1000))
            writer.add_images("x", imgs_x, int(idx_t / 1000))
            writer.add_images("mu_img", imgs_mu_img, int(idx_t / 1000))
            writer.add_images("z_1_img", z_img, int(idx_t / 1000))
            writer.add_images("z_sum_img", z_img_sum, int(idx_t / 1000))
            writer.add_images("z0_img", z0_img, int(idx_t / 1000))

            evaluate(model, test_loader, int(idx_t / 1000), writer, device)  # test KODAK dataset

        idx_t += 1

    return idx_t


def evaluate(model, test_loader, idx, writer, device):

    model.eval()
    with torch.no_grad():

        distortion = 0
        compression = 0

        for images, _ in test_loader:

            accuracy, k, x_hat, _, z_hat, _, _ = model.get_accuracy(images.to(device))
            z_size = torch.sum(k.detach() * model.n, (1, 2)).cpu()
            compression += bpp(k.shape[1], k.shape[2], model.n, torch.mean(z_size).item(), 512 * 768, np.log2(model.L))

            distortion += torch.mean(accuracy).cpu().clone().numpy()

        writer.add_scalar('test_mean_accuracy', distortion, idx)
        writer.add_scalar('test_mean_compression', compression, idx)

'''
    this function tests the model on KODAK and plots a scatter plot with accuracy and compression for each KODAK image
'''
def print_RD_curve(model, test_loader, idx_v, writer, device):

    model.eval()
    with torch.no_grad():

        for images, _ in test_loader:

            accuracy, k, x_hat, z0, z_hat, mu, a = model.get_accuracy(images.to(device))
            z_size = torch.sum(k.detach() * model.n, (1, 2)).cpu()
            compression = bpp(k.shape[1], k.shape[2], model.n, z_size.numpy(), 512 * 768, np.log2(model.L))

            distortion = accuracy.cpu().clone().numpy()

        idx_sort = np.argsort(compression)
        distortion = distortion[idx_sort]
        compression = compression[idx_sort]

        fig = plt.figure()
        c_max = np.max(compression)*1.2 if np.max(compression) > 1. else 1.
        d_min = np.min(distortion)*0.97 if np.min(distortion) < 85.0 else 85.0
        plt.axis([0, c_max, d_min, 100.0])
        plt.scatter(compression, distortion)
        plt.grid()

        writer.add_figure("rate_distortion_KODAK", fig, idx_v)

        img = x_hat[0].detach().cpu().clone().numpy()
        mu_img = mu.detach().cpu().clone().numpy()
        mu_img = mu_img / np.expand_dims(np.expand_dims(np.max(mu_img, (1, 2)), -1), -1)
        img_mu_img = (np.expand_dims(mu_img, 1))[0]
        z0_img = z0[0:3].detach().cpu().clone().numpy()
        writer.add_image("KODAK", img, idx_v)
        writer.add_image("KODAK_mu_img", img_mu_img, idx_v)
        writer.add_images("KODAK_z0_img", z0_img, idx_v)

        idx_v += 1

    return idx_v


def main():

    print(sys.argv[0])
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])

    torch.manual_seed(1234)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if sys.argv[1] == "khazadum":
        train_folder = "/local_storage/datasets/imagenet/train"
    elif sys.argv[1] == "rivendell":
        train_folder = "/local_storage/datasets/ILSVRC2015/Data/CLS-LOC/train"
    else:
        print("Error, wrong node")
        return

    print("training on " + sys.argv[1] + " on device: " + str(device))

    lr_d = 0.0003  # learning rate for the main optimizer
    # one idea was to divide the to losses into two different optimization steps
    lr_k = 0.0003
    second_optimizer = True

    gamma = float(sys.argv[2]) #0.7  # learning rate decay # TODO: we can try different values
    lr_step_size = 1  # decay every epoch
    EPOCHS = 10  # TODO: maybe more?
    Batch_size = 32
    min_accuracy = 97.0  # min error before switching to compression optimization
    colors = 3
    model_depth = int(sys.argv[3])  # TODO: either 3 or 5
    model_size = 64  # TODO: either 64 or 128
    n = 64
    HW = 168
    L2 = 0.0  # for now no regularization, we can make experiments on this
    L = 8  # number of symbols used used by the quantization step
    L2_a = 0.0000005  # regularization on mu  # TODO:performances sensible to this parameter
    beta = 0.1  # L_tot = L1 + beta * L2  # TODO: performances sensible to this parameter

    ''' MODEL DEFINITION '''

    model = network.Net(min_accuracy, colors, model_depth, model_size, n, L, device).to(device)

    ''' DATASET LOADER '''
    trans_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.Resize(256),
                                      transforms.CenterCrop(HW),
                                      transforms.ToTensor()])

    trans_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(root=train_folder, transform=trans_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=8)

    # train_dataset = ImageNet(root='/local_storage/datasets/imagenet', split='train', transform=trans_train, download=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=8)

    test_dataset = datasets.ImageFolder(root="/local_storage/datasets/KODAK_padded", transform=trans_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=24, shuffle=True, num_workers=8)

    ''' TENSORBOARD WRITER '''

    #/Midgard/home/areichlin/compression
    log_dir = './debug_log/two_opt_adam_z_detach_gamma_'+str(gamma)+'_depth_'+str(model_depth)
    writer = SummaryWriter(log_dir=log_dir)

    ''' OPTIMIZER, SCHEDULER DEFINITION '''

    optimizer_d = optim.Adam(model.parameters(), lr=lr_d, weight_decay=L2)
    optimizer_k = optim.Adam(model.parameters(), lr=lr_k)

    scheduler_d = StepLR(optimizer_d, step_size=lr_step_size, gamma=gamma)

    idx_t = 0
    idx_v = 0

    ''' TRAINING LOOP '''

    for epoch in range(1, EPOCHS + 1):
        idx_t = train(model, train_loader, test_loader, optimizer_d, optimizer_k, second_optimizer, L2_a, beta, idx_t, writer, device)
        scheduler_d.step()
        idx_v = print_RD_curve(model, test_loader, idx_v, writer, device)

    writer.close()

    # TODO: add saving parameters of the best model


if __name__ == '__main__':
    main()
