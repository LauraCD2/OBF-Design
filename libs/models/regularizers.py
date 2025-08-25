import torch

def num_band_reg(mask, N=500):
    # Number of bands regularization
    # mask: binary mask with shape (n_bands)
    # N: number of bands
    Nest = torch.sum(mask)
    return torch.abs((N - Nest) / N)


def binary_reg(mask):
    # Binary regularization
    # mask: binary mask with shape (n_bands)
    out = torch.square(mask - 1) * torch.square(mask)
    return out.mean()
