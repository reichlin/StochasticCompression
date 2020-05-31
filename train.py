import time
import torch
import torch.nn as nn
from tqdm import tqdm
import network as network
import matplotlib as mpl
from utils import *
mpl.use('Agg')
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, optimizer_d, optimizer_k, second_optimizer, L2_a, beta, clip_gradient, idx_t, writer, device, distortion_training_epochs):

    for i, (images, _) in tqdm(enumerate(train_loader)):

        model.train()

        gpu_imgs = images.to(device).detach()

        x_hat_compress, x_hat_k, log_pk, log_p_compression, k, k_compression, mu, a, z = model(gpu_imgs)
        loss_d, accuracy_k, accuracy_c, accuracy_compression_mean, accuracy_k_mean, img_err_k, img_err_c = model.get_loss_d(gpu_imgs, x_hat_compress, x_hat_k)

        if i % distortion_training_epochs == 0:

            loss_k, avg_cond, R = model.get_loss_k(gpu_imgs, img_err_k, img_err_c, accuracy_k, accuracy_c, k, k_compression, log_pk, log_p_compression, a, L2_a)

            loss = loss_d + beta * loss_k

            optimizer_d.zero_grad()
            if clip_gradient:
                torch.nn.utils.clip_grad_norm(model.parameters(), 10.)
            loss.backward(retain_graph=False)
            optimizer_d.step()

        else:
            loss = loss_d

            optimizer_d.zero_grad()
            if clip_gradient:
                torch.nn.utils.clip_grad_norm(model.parameters(), 10.)
            loss.backward(retain_graph=False)
            optimizer_d.step()

        ''' bunch of constants for the tensorboard '''
        z_size = torch.sum(k.detach() * model.n, (1, 2)).cpu()
        z_size_mean = torch.mean(z_size).item()
        z_size_max = torch.max(z_size).item()
        z_size_min = torch.min(z_size).item()
        accuracy_k_mean = accuracy_k_mean.item()
        accuracy_compression_mean = accuracy_compression_mean.item()
        if i % distortion_training_epochs == 0:
            avg_cond = avg_cond.item()
            R = torch.mean(R).item()
        mu_img = mu.detach().cpu().clone().numpy()
        mu_img = mu_img / np.expand_dims(np.expand_dims(np.max(mu_img, (1, 2)), -1), -1)
        a = torch.mean(a).item()
        c_span = torch.abs(torch.max(model.c.detach().cpu()) - torch.min(model.c.detach().cpu())).item()
        k_values = torch.flatten(k).detach().cpu()

        ''' compute the bit per pixels of the images '''
        bpp_mean = bpp(z_size_mean, model.L, k.shape[1], k.shape[2], model.n, images.shape[-1], images.shape[-2])
        bpp_max = bpp(z_size_max, model.L, k.shape[1], k.shape[2], model.n, images.shape[-1], images.shape[-2])
        bpp_min = bpp(z_size_min, model.L, k.shape[1], k.shape[2], model.n, images.shape[-1], images.shape[-2])

        writer.add_scalar('training_accuracy', accuracy_k_mean, idx_t)
        writer.add_scalar('training_accuracy_compression', accuracy_compression_mean, idx_t)
        writer.add_scalar('compression_average', bpp_mean, idx_t)
        writer.add_scalar('compression_max', bpp_max, idx_t)
        writer.add_scalar('compression_min', bpp_min, idx_t)
        writer.add_scalar('span_centroids', c_span, idx_t)
        if i % distortion_training_epochs == 0:
            writer.add_scalar('cond_average', avg_cond, idx_t)
            writer.add_scalar('R_value_average', R, idx_t)
        writer.add_scalar('mu_value_average', a, idx_t)

        if idx_t % 1000 == 0 and idx_t != 0:

            writer.add_histogram('k', k_values, idx_t)

            imgs_x = images[0:3].detach().clone().numpy()
            imgs_x_hat = x_hat_k[0:3].detach().cpu().clone().numpy()
            imgs_mu_img = (np.expand_dims(mu_img, 1))[0:3]
            writer.add_images("x_hat", imgs_x_hat, int(idx_t / 1000))
            writer.add_images("x", imgs_x, int(idx_t / 1000))
            writer.add_images("mu_img", imgs_mu_img, int(idx_t / 1000))

            evaluate(model, test_loader, int(idx_t / 1000), writer, device)  # test KODAK dataset

        idx_t += 1

    return idx_t


def evaluate(model, test_loader, idx, writer, device):

    model.eval()
    with torch.no_grad():

        distortion = []
        compression = []

        bins = []
        for i in range(model.n + 1):
            bins.append(i)

        for images, _ in test_loader:

            accuracy, x_hat, mu = model.get_accuracy(images.to(device))

            z_size = torch.sum(mu.detach() * model.n, (1, 2)).cpu()
            compression.append(bpp(torch.mean(z_size).item(), model.L, mu.shape[1], mu.shape[2], model.n, 512., 768.))
            distortion.append(torch.mean(accuracy).cpu().clone().numpy())

        d = np.average(distortion)
        c = np.average(compression)

        mu_values_n = torch.flatten(torch.ceil(mu * model.n)).detach().cpu().numpy()
        fig = plt.figure()
        plt.hist(mu_values_n, bins=bins)
        plt.grid()
        writer.add_figure("histogram_mu", fig, idx)

        x = np.linspace(0.05, 4., 1000)
        min_dist = np.square(x[0] - c)+np.square((1. - 1. / (108. * x[0])) - (d/100.))
        for i in range(len(x)):
            dist = np.square(x[i] - c) + np.square((1. - 1. / (108. * x[i])) - (d/100.))
            if dist < min_dist:
                min_dist = dist
        min_dist = np.sqrt(min_dist)

        writer.add_scalar('test_mean_accuracy', d, idx)
        writer.add_scalar('test_mean_compression', c, idx)
        writer.add_scalar('test_min_distance', min_dist, idx)


def test_channels_order(model, test_loader, idx_v, writer, device):

    model.eval()
    with torch.no_grad():

        for images, _ in test_loader:

            acc_channels, acc_channels_cumulative_z = model.get_information_content(images.to(device))

            fig = plt.figure()
            plt.plot(np.linspace(1, model.n, model.n), acc_channels)
            plt.grid()
            writer.add_figure("average_information_content", fig, idx_v)

            fig = plt.figure()
            plt.plot(np.linspace(1, model.n, model.n), acc_channels_cumulative_z)
            plt.grid()
            writer.add_figure("cumulative_information_content", fig, idx_v)


def print_RD_curve(model, test_loader, idx_v, writer, device):

    model.eval()
    with torch.no_grad():

        distortion = []
        compression = []

        for images, _ in test_loader:

            accuracy, x_hat, mu = model.get_accuracy(images.to(device))
            z_size = torch.sum(mu.detach() * model.n, (1, 2)).cpu()
            compression.append(bpp(torch.mean(z_size).item(), model.L, mu.shape[1], mu.shape[2], model.n, 512., 768.))

            distortion.append(torch.mean(accuracy).cpu().clone().item())

        fig = plt.figure()
        c_max = np.max(compression)*1.2 if np.max(compression) > 1. else 1.
        d_min = np.min(distortion)*0.97 if np.min(distortion) < 85.0 else 85.0
        plt.axis([0, c_max, d_min, 100.0])
        plt.scatter(compression, distortion)
        plt.grid()

        writer.add_figure("rate_distortion_KODAK", fig, idx_v)

        img = x_hat[0].detach().cpu().clone().numpy()
        img_bar = np.zeros([img.shape[0], img.shape[1] + 1, img.shape[2]])
        compression_bar = np.zeros(img.shape[2])
        for i in range(int(np.clip(compression[0], 0, 1.0) * img.shape[2])):
            compression_bar[i] = 1.0
        img_bar[:, :-1, :] = img[:, :, :]
        img_bar[:, -1, :] = compression_bar

        mu_img = mu.detach().cpu().clone().numpy()
        mu_img = mu_img / np.expand_dims(np.expand_dims(np.max(mu_img, (1, 2)), -1), -1)
        img_mu_img = (np.expand_dims(mu_img, 1))[0]
        writer.add_image("KODAK", img_bar, idx_v)
        writer.add_image("KODAK_mu_img", img_mu_img, idx_v)

        idx_v += 1

    return idx_v


def main():

    torch.manual_seed(1234)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_backend = torch.cuda.is_available()

    if sys.argv[1] == "khazadum":
        train_folder = "/local_storage/datasets/imagenet/train"
    elif sys.argv[1] == "rivendell":
        train_folder = "/local_storage/datasets/ILSVRC2015/Data/CLS-LOC/train"
    else:
        print("Error, wrong node")
        return

    print("training on " + sys.argv[1] + " on device: " + str(device))


    ''' PARAMETERS '''

    # AUTOENCODER
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
    distortion_training_epochs = 1 #int(sys.argv[2])  # {1, 2, 5}

    # POLICY SEARCH

    k_sampling_policy = 1 #int(sys.argv[3])  # 0:binary, 1:poisson
    exploration_epsilon = 0.0

    compression_sampling_function = 2 # int(sys.argv[4])  # 0:U*mu+0.2, 1:Exponential, 2:Pareto Bounded
    adaptive_compression_sampling = True#int(sys.argv[5]) == 1  # {False, True}

    pareto_alpha = 1.16#float(sys.argv[2])
    pareto_interval = 0.5#float(sys.argv[3])


    ''' MODEL DEFINITION '''
    model = network.Net(min_accuracy,
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
                        pareto_alpha).to(device)

    ''' DATASET LOADER '''
    trans_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.Resize(256),
                                      transforms.CenterCrop(HW),
                                      transforms.ToTensor()])

    trans_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(root=train_folder, transform=trans_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=8)

    test_dataset = datasets.ImageFolder(root="/local_storage/datasets/KODAK", transform=trans_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=8)
    test_dataset_padded = datasets.ImageFolder(root="/local_storage/datasets/KODAK_padded", transform=trans_test)
    test_loader_padded = DataLoader(dataset=test_dataset_padded, batch_size=24, shuffle=True, num_workers=8)

    ''' TENSORBOARD WRITER '''

    name = 'Bounded_Pareto_adaptive' #_alpha_'+str(pareto_alpha)+'_delta_'+str(pareto_interval)
    log_dir = '/Midgard/home/areichlin/compression/pareto_experiments/'+name
    writer = SummaryWriter(log_dir=log_dir)

    ''' OPTIMIZER, SCHEDULER DEFINITION '''

    optimizer_d = optim.Adam(model.parameters(), lr=lr_d, weight_decay=L2)
    optimizer_k = optim.Adam(model.parameters(), lr=lr_k)

    scheduler_d = StepLR(optimizer_d, step_size=lr_step_size, gamma=gamma)

    idx_t = 0
    idx_v = 0

    ''' TRAINING LOOP '''

    print("start training")

    for epoch in range(1, EPOCHS + 1):
        idx_t = train(model,
                      train_loader,
                      test_loader,
                      optimizer_d,
                      optimizer_k,
                      second_optimizer,
                      L2_a,
                      beta,
                      clip_gradient,
                      idx_t,
                      writer,
                      device,
                      distortion_training_epochs)
        scheduler_d.step()
        test_channels_order(model, test_loader_padded, idx_v, writer, device)
        idx_v = print_RD_curve(model, test_loader, idx_v, writer, device)
        print("saving model")
        torch.save(model.state_dict(), '/Midgard/home/areichlin/compression/models/' + name + '_' + str(epoch) + '.pt')

    writer.close()


if __name__ == '__main__':
    main()
