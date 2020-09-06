import sys

import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from network import Net
import transformer_utils as utils
from transformer import CompressionTransformer, TransformerLoss


def evaluate(net, transformer, test_loader, idx_t, writer, device):

    transformer.evaluate()

    with torch.no_grad():

        symbols = []
        compression = []
        distortion_k = []
        distortion_zk = []

        for i, (images, _) in enumerate(test_loader):

            gpu_images = images.to(device).detach()

            z, k = net.E(gpu_images)
            z = net.quantize(z)
            z_k = net.mask_z(z, k)

            output = transformer(z_k, training=False)

            c_k = output['c_k']
            mewtwo = output['mewtwo']
            z_rec_k = output['z_rec_k']

            mask = (z_k == 0)
            _, _, h, w = gpu_images.shape
            _, zc, zh, zw = z_k.shape

            avg_bpp = utils.bpp_transformer(mewtwo.mean().item(), zh, zw, zc, h, w)
            avg_num_symbols = (c_k != 0).sum(dtype=torch.float).detach() / c_k.shape[1]

            # only look at the unmasked positions in z to calc. accuracy
            accuracy_zk = (z_k == z_rec_k).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())

            x_hat_rec_k = net.D(z_rec_k * mask)

            if net.cuda_backend:
                x_uint8 = ((images * 255).type(torch.cuda.IntTensor)).type(torch.cuda.FloatTensor)
            else:
                x_uint8 = ((images * 255).type(torch.IntTensor)).type(torch.FloatTensor)

            x_hat_rec_k_uint8 = (x_hat_rec_k * 255)

            accuracy_k = 100 * net.msssim_test(x_uint8, x_hat_rec_k_uint8)

            symbols.append(avg_num_symbols)
            compression.append(avg_bpp)
            distortion_k.append(accuracy_k.cpu().clone().numpy())
            distortion_zk.append(accuracy_zk.cpu().clone().numpy())

        symbols = np.mean(symbols)
        compr = np.mean(compression)
        acc_k = np.mean(distortion_k)
        acc_zk = np.mean(distortion_zk)

        writer.add_scalar('testing_compression', compr)
        writer.add_scalar('testing_num_symbols', symbols)
        writer.add_scalar('testing_accuracy_k', acc_k.item(), idx_t)
        writer.add_scalar('testing_accuracy_zk', acc_zk.item(), idx_t)

    transformer.train()


def train_one_epoch(net, transformer, train_loader, test_loader, optimizer, loss_func, gamma, idx_t, writer, device):

    for i, (images, _) in enumerate(train_loader):

        gpu_images = images.to(device).detach()

        out = net(gpu_images)  # out = x_hat_compress, x_hat_k, log_pk, k, k_compression, mu, z_k
        z_k = out[-1].detach()  # b x d x h x w
        x_hat_k = out[1].detach()

        output = transformer(z_k)

        k = output['k']
        k_compress = output['k_compress']
        c_k = output['c_k']
        log_pk = output['log_pk']
        mewtwo = output['mewtwo']
        z_rec_k = output['z_rec_k']
        z_rec_compress = output['z_rec_compress']
        z_logits_k = output['z_logits_k']
        z_logits_compress = output['z_logits_compress']

        loss_ce, loss_pg, policy_loss = loss_func(z_k, z_rec_compress, z_logits_compress, k, log_pk)

        loss = loss_ce + gamma * loss_pg

        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        '''
        Computing some metrics
        '''

        mask = (z_k == 0)
        _, _, h, w = gpu_images.shape
        _, zc, zh, zw = z_k.shape

        avg_bpp = utils.bpp_transformer(mewtwo.mean().item(), zh, zw, zc, h, w)
        min_bpp = utils.bpp_transformer(mewtwo.min().item(), zh, zw, zc, h, w)
        max_bpp = utils.bpp_transformer(mewtwo.max().item(), zh, zw, zc, h, w)
        avg_num_symbols = (c_k != 0).sum(dtype=torch.float).detach() / c_k.shape[1]

        # only look at the unmasked positions in z to calc. accuracy
        accuracy_zk = (z_k == z_rec_k).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())
        accuracy_zcompress = (z_k == z_rec_compress).sum(dtype=torch.float) / (mask.shape.numel() - mask.sum())

        x_hat_rec_k = net.D(z_rec_k)
        x_hat_rec_compress = net.D(z_rec_compress)

        if net.cuda_backend:
            x_uint8 = ((images * 255).type(torch.cuda.IntTensor)).type(torch.cuda.FloatTensor)
        else:
            x_uint8 = ((images * 255).type(torch.IntTensor)).type(torch.FloatTensor)

        x_hat_k_uint8 = (x_hat_k * 255)
        x_hat_rec_k_uint8 = (x_hat_rec_k * 255)
        x_hat_rec_compress_uint8 = (x_hat_rec_compress * 255)

        accuracy_no_transformer = 100 * net.msssim_test(x_uint8, x_hat_k_uint8)
        accuracy_k = 100 * net.msssim_test(x_uint8, x_hat_rec_k_uint8)
        accuracy_compress = 100 * net.msssim_test(x_uint8, x_hat_rec_compress_uint8)

        writer.add_scalar('k mean', k.mean().item(), idx_t)
        writer.add_scalar('k_compress mean', k_compress.mean().item(), idx_t)
        writer.add_scalar('mewtwo mean', mewtwo.mean().item(), idx_t)
        writer.add_scalar('loss', loss.item(), idx_t)
        writer.add_scalar('loss_ce', loss_ce.item(), idx_t)
        writer.add_scalar('loss_pg', loss_pg.item(), idx_t)
        writer.add_histogram('c_k', c_k.detach().flatten(0), idx_t)
        writer.add_scalar('training_accuracy_no_transformer', accuracy_no_transformer.item(), idx_t)
        writer.add_scalar('training_accuracy_k', accuracy_k.item(), idx_t)
        writer.add_scalar('training_accuracy_compress', accuracy_compress.item(), idx_t)
        writer.add_scalar('training_accuracy_zk', accuracy_zk.item(), idx_t)
        writer.add_scalar('training_accuracy_zcompress', accuracy_zcompress.item(), idx_t)
        writer.add_scalar('policy_gradient_loss', int(policy_loss), idx_t)
        writer.add_scalar('avg_bpp', avg_bpp, idx_t)
        writer.add_scalar('min_bpp', min_bpp, idx_t)
        writer.add_scalar('max_bpp', max_bpp, idx_t)
        writer.add_scalar('avg_num_symbols', avg_num_symbols, idx_t)

        if idx_t % 1000 == 0 and idx_t != 0:

            imgs_x = images[0:3].detach().clone().numpy()
            imgs_x_hat = x_hat_k[0:3].detach().cpu().clone().numpy()
            imgs_x_hat_rec_k = x_hat_rec_k[0:3].detach().cpu().clone().numpy()
            imgs_x_hat_rec_compress = x_hat_rec_compress[0:3].detach().cpu().clone().numpy()
            writer.add_images("x_hat_transf_k", imgs_x_hat_rec_k, int(idx_t / 1000))
            writer.add_images("x_hat_transf_compress", imgs_x_hat_rec_compress, int(idx_t / 1000))
            writer.add_images("x_hat_r", imgs_x_hat, int(idx_t / 1000))
            writer.add_images("x_hat", imgs_x_hat, int(idx_t / 1000))
            writer.add_images("x", imgs_x, int(idx_t / 1000))

            evaluate(net, transformer, test_loader, int(idx_t/1000), writer, device)

        idx_t += 1

    return idx_t


def main():

    if len(sys.argv) == 1:
        print('Local run!')
        threshold = 0.95
        gamma = 0.1
        batch_size = 4
        local = True
    else:
        print('Running script @' + sys.argv[1])
        threshold = float(sys.argv[2])
        gamma = float(sys.argv[3])
        batch_size = 32  # the whole test set
        local = False

    torch.manual_seed(1234)
    cuda_backend = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_backend else "cpu")

    ############# Alfredo help me out :) #############

    if local:
        test_folder = '../Kodak/'
        path_to_model = '../models/accuracy_97.597206.pt'
        logdir = './runs/'
    else:
        test_folder = '/local_storage/datasets/KODAK'
        path_to_model = '/Midgard/home/arechlin/compression/models/best_final_model_97.6.pt'
        logdir = '/Midgard/home/areichlin/compression/alex/transformer/logs/'

        if sys.argv[1] == "khazadum":
            train_folder = "/local_storage/datasets/imagenet/train"
        elif sys.argv[1] == "rivendell":
            train_folder = "/local_storage/datasets/ILSVRC2015/Data/CLS-LOC/train"
        else:
            print("Error, wrong node")
            return

    ##################################################

    trans_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.Resize(256),
                                      transforms.CenterCrop(168),
                                      transforms.ToTensor()])

    trans_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(root=train_folder, transform=trans_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = datasets.ImageFolder(root=test_folder, transform=trans_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=1)

    '''
    Encoder Decoder parameters
    '''

    L = 8
    n = 64
    colors = 3
    model_depth = 3
    model_size = 64
    min_accuracy = 97.0
    pareto_alpha = 1.16
    pareto_interval = 0.5

    net = Net(min_accuracy, colors, model_depth, model_size, n, L,
              cuda_backend, pareto_interval, pareto_alpha).to(device)

    net.load_state_dict(torch.load(path_to_model, map_location=device), strict=False)

    z_centroids = net.c.detach()
    loss = TransformerLoss(z_centroids, threshold=threshold)

    '''
    Transformer Parameters
    '''

    nhead = 4
    d_ff = 2048
    dropout = 0.1
    num_layers = 4
    num_centroids = 8
    num_enc_layers = num_dec_layers = num_layers
    pareto_interval_transf = 0.3  # TODO: Test different values

    model = CompressionTransformer(model_size, nhead, d_ff, dropout, num_centroids,
                                   num_enc_layers, num_dec_layers, z_centroids, pareto_alpha,
                                   pareto_interval_transf, cuda_backend).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0003)

    model_name = f'transformer_gamma_{gamma}_threshold_{threshold}'
    writer = SummaryWriter(logdir + model_name)

    EPOCHS = 10
    for epoch in range(EPOCHS):

        train_one_epoch(net=net,
                        transformer=model,
                        )

    if local:
        save_path = './models/' + model_name + '.pt'
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()