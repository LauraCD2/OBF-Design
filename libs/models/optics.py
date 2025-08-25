import torch
from torch import nn
from torch.autograd import Function, Variable

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = ( torch.sign(input) + 1 ) / 2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.clamp(0, 1)  # Ensure the gradient is between 0 and 1
        return grad_input


class BinaryBandSelection(nn.Module):
    # Metodo de Kebin/Seismic
    def __init__(self, n_bands, model: nn.Module):
        super(BinaryBandSelection, self).__init__()

        self.mask = nn.Parameter(torch.randn(n_bands), requires_grad=True)
        self.model = model

    def forward(self, x):
        # x with shape (batch, n_bands, n_features)
        mask = BinaryQuantize.apply(self.mask)

        mask = mask[None, ...]  # (1, n_bands)
        if x.dim() == 3:
            mask = mask[..., None]  # (1, n_bands, 1)

        x = x * mask
        return self.model(x)

    def get_binary_mask(self):
        mask = BinaryQuantize.apply(self.mask)
        return mask


class BandSelection(nn.Module):
    # Metodo de Karen
    def __init__(self, n_bands, model: nn.Module, learned_bands):
        super(BandSelection, self).__init__()

        self.mask = nn.Parameter(torch.rand(n_bands), requires_grad=True)
        self.model = model
        self.binarize = False
        self.learned_bands = learned_bands

    def forward(self, x):
        # x with shape (batch, n_bands, n_features)
        if self.binarize:
            # select k-top bands
            indx = torch.topk(self.mask, k=self.learned_bands, dim=-1).indices
            mask = torch.zeros_like(self.mask)
            mask[indx] = 1.0
        else:
            mask = self.mask

        mask = mask[None, ...]  # (1, n_bands)
        if x.dim() == 3:
            mask = mask[..., None]  # (1, n_bands, 1)

        x = x * mask
        return self.model(x)

    def get_binary_mask(self):
        indx = torch.topk(self.mask, k=self.learned_bands, dim=-1).indices
        mask = torch.zeros_like(self.mask)
        mask[indx] = 1.0
        return mask


class FilterDesign(nn.Module):
    def __init__(self, n_bands, n_filters, model: nn.Module):
        super(FilterDesign, self).__init__()
        self.mu = nn.Parameter(torch.linspace(0.2, 0.8, n_filters), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(n_filters) * 0.02, requires_grad=True)
        self.samples = torch.linspace(0, 1, n_bands)
        self.model = model
        # minimun and maximum FWHM (Full Width at Half Maximum) values
        self.max_val = torch.tensor(0.05)
        self.min_val = torch.tensor(0.01) 
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma = 2.3548 * sigma
        # For cocoa dataset the spectral range is 500 - 850 nm
        # interval of 350 nm
        # min FWHM =  350 * 2.3548 * 0.01 = 8.25 nm
        # max FWHM =  350 * 2.3548 * 0.05 = 41.25 nm

    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * torch.square((x - mu) / sigma))

    def get_filters(self):
        samples = self.samples[..., None].to(self.mu.device)
        sigmas = torch.clamp(
            self.sigma[None, ...],
            self.min_val.to(self.mu.device),
            self.max_val.to(self.mu.device),
        )
        filters = self.gaussian(samples, self.mu[None, ...], sigmas)
        return filters / filters.sum(dim=0, keepdim=True)

    def filter_params(self):
        # Get the parameters of the filters
        mu = self.mu.detach().cpu().numpy()
        sigma = self.sigma.detach().cpu().numpy()
        return mu, sigma

    def forward(self, x):
        return self.model(x @ self.get_filters())