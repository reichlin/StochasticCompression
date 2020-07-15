import torch
import numpy as np
import matplotlib.pyplot as plt


def bpp_transformer(mu_mean, symbols=8, zh=21, zw=21, zc=64, h=168, w=168):
    """ Defaults are for Kodak testset """
    return np.log2(symbols) * mu_mean * zh * zw * zc / (h * w)


def generate_subsequent_mask(seq_len):
    """
    Generate an autoregressive mask for the sequence. The masked positions are filled with float('-inf')
    and unmasked positions with float(0.0).
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_non_autoregressive_mask(seq_len):
    """
    Generates a mask for tgt to only attend to its own position when decoding. Masked positions are filled
    with float('-inf'), and unmasked positions with float(0.0).
    """
    mask = torch.eye(seq_len)
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def flatten_tensor_batch(tensor):
    """
    Flattens out a batch of images to be on the transformer format:
        B x D x H x W --> HW x B x D
    """
    return tensor.flatten(2).permute(2, 0, 1)


def unravel_tensor_batch(flat_tensor, original_shape):
    """
    Unravels a flattened batch of images, form transformer format to standard torch format:
        HW x B x D --> B x D x H x W
    """
    b, d, h, w = original_shape
    return flat_tensor.permute(1, 2, 0).view(b, d, h, w)


def flatten_mask(mask):
    """
    Flattens the mask:
        B x H x W --> B x HW
    """
    return mask.flatten(1)


def mask_output(out, mask):
    """
    Apply the padding mask to output of transformer to keep original sizes.
        out ~ B x D x H x W
        mask ~ B x H x W, with True/1 in zero-padded pixels, and False/0 in actual pixels.
    """
    return out * (~mask.to(torch.bool).unsqueeze(1).repeat(1, 3, 1, 1))


def show_img(img):
    plt.imshow(img.permute(1, 2, 0).detach().numpy())
    plt.show()


if __name__ == '__main__':

    b = 2
    d = 4
    h = 3
    w = 3

    x = torch.rand(b, d, h, w)
    x_flat = flatten_tensor_batch(x)

    mask = torch.zeros(b, h, w)
    mask_flat = flatten_mask(mask)

    a = 0
