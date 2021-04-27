""" Special layers to incapsulate attribute computation into neural network. """
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from ..batchflow.models.torch import ResBlock

class InstantaneousPhaseLayer(nn.Module):
    """ Instantaneous phase comnputation. """
    def __init__(self, inputs=None, continuous=False, **kwargs):
        super().__init__()
        self.continuous = continuous

    def _hilbert(self, x):
        """ Hilbert transformation. """
        # import pdb; pdb.set_trace()
        N = x.shape[-1]
        # x = torch.stack([x, torch.zeros_like(x)], dim=-1)
        # fft = torch.fft(x, signal_ndim=1, normalized=False)
        fft = torch.fft.fft(x)

        h = torch.zeros(N, device=x.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        if x.ndim > 1:
            shape = [1] * x.ndim
            # shape[-2] = N
            shape[-1] = N
            h = h.view(*shape)

        # result = torch.ifft(fft * h, signal_ndim=1)
        result = torch.fft.ifft(fft * h)
        return result

    def _angle(self, x):
        """ Compute angle of complex number. """
        # res = torch.atan(x[..., 1] / x[..., 0])
        res = torch.atan(x.imag / x.real)
        res[x.real == 0] = np.pi
        res = res % (2 * np.pi) - np.pi
        if self.continuous:
            res =  torch.abs(res) - np.pi / 2
        return res

    def forward(self, x):
        """ Forward pass. """
        x = self._hilbert(x)
        x = self._angle(x)
        return x

class MovingNormalizationLayer(nn.Module):
    def __init__(self, inputs=None, window=(1, 1, 100), padding='same', fill_value=0):
        super().__init__()
        self.window = window
        self.padding = padding
        self.fill_value = fill_value
        self.ndim = inputs.ndim
        self.kernel = torch.ones((1, 1, *window), dtype=torch.float32, requires_grad=False).to(inputs.device)

    def forward(self, x):
        ndim = x.ndim
        device = x.device
        window = self.window
        x = expand_dims(x)
        pad = [(w // 2, w - w // 2 - 1) for w in self.window]
        if self.padding == 'same':
            num = F.pad(x, (*pad[2], *pad[1], *pad[0], 0, 0, 0, 0))
        else:
            num = x
            # n = torch.ones_like(x, dtype=torch.float32, device=device, requires_grad=False)
            # n = F.conv3d(n, torch.ones((1, 1, *window), dtype=torch.float32, requires_grad=False).to(device))
        n = np.prod(self.window)
        num = (num - num.min()) / (num.max() - num.min())
        mean = F.conv3d(num, self.kernel / n)
        mean_2 = F.conv3d(num ** 2, self.kernel / n)
        std = torch.clip(mean_2 - mean ** 2, min=0) ** 0.5
        print(x.shape, x.min(), x.max(), std.min(), std.max())
        if self.padding == 'valid':
            x = x[:, :, pad[0][0]:x.shape[2]-pad[0][1], pad[1][0]:x.shape[3]-pad[1][1], pad[2][0]:x.shape[4]-pad[2][1]]
        result = torch.nan_to_num((x - mean) / std, nan=self.fill_value)
        return squueze(result, self.ndim)

class SemblanceLayer(nn.Module):
    def __init__(self, inputs=None, window=(1, 5, 20), fill_value=1):
        super().__init__()
        self.ndim = inputs.ndim
        self.window = window
        self.fill_value = fill_value

    def forward(self, x):
        device = x.device
        window = self.window
        x = expand_dims(x)

        padding = [(w // 2, w - w // 2 - 1) for w in window]
        num = F.pad(x, (0, 0, *padding[1], *padding[0], 0, 0, 0, 0))
        num = F.conv3d(num, torch.ones((1, 1, window[0], window[1], 1), dtype=torch.float32, requires_grad=False).to(device)) ** 2
        num = F.pad(num, (*padding[2], 0, 0, 0, 0, 0, 0, 0, 0))
        num = F.conv3d(num, torch.ones((1, 1, 1, 1, window[2]), dtype=torch.float32, requires_grad=False).to(device))

        denum = F.pad(x, (*padding[2], *padding[1], *padding[0], 0, 0, 0, 0))
        denum = F.conv3d(denum ** 2, torch.ones((1, 1, *window), dtype=torch.float32, requires_grad=False).to(device))

        normilizing = torch.ones(x.shape[:-1], dtype=torch.float32, requires_grad=False).to(device)
        normilizing = F.pad(normilizing, (*padding[1], *padding[0], 0, 0, 0, 0))
        normilizing = F.conv2d(normilizing, torch.ones((1, 1, window[0], window[1]), dtype=torch.float32, requires_grad=False).to(device))

        denum *= normilizing.view(*normilizing.shape, 1)

        result = torch.nan_to_num(num / denum, nan=self.fill_value)

        return squueze(result, self.ndim)

class FrequenciesFilterLayer(nn.Module):
    def __init__(self, inputs, q=0.1, window=200):
        super().__init__()
        self.q = q
        self.window = window

    def forward(self, inputs):
        inputs = inputs.view(-1, inputs.shape[-1])
        sfft = torch.stft(inputs, self.window, return_complex=True)
        q_ = int(sfft.shape[-2] * self.q)
        sfft[:, :q_] = 0
        sfft[:, -q_:] = 0
        return torch.istft(sfft, self.window).view(*inputs.shape)

def expand_dims(x):
    if x.ndim == 4:
        x = x.view(x.shape[0], 1, *x.shape[-3:])
    elif x.ndim == 3:
        x = x.view(1, 1, *x.shape)
    elif x.ndim == 2:
        x = x.view(1, 1, 1, *x.shape)
    return x

def squueze(x, ndim):
    if ndim == 4:
        return x[:, 0]
    if ndim == 3:
        return x[0, 0]
    if ndim == 2:
        return x[0, 0, 0]
    return x

class InputLayer(nn.Module):
    """ Input layer with possibility of instantaneous phase concatenation. """
    def __init__(self, inputs, normalization=False, phases=False, continuous=False,
                 window=100, base_block=ResBlock, **kwargs):
        super().__init__()
        self.normalization = normalization
        self.phases = phases
        if self.normalization:
            self.normalization_layer = MovingNormalizationLayer(inputs, window)
        if self.phases:
            self.phase_layer = InstantaneousPhaseLayer(continuous)
            phases_ = self.phase_layer(inputs)
            inputs = self._concat(inputs, phases_)
        self.base_block = base_block(inputs, **kwargs)

    def _concat(self, x, phases):
        x = torch.cat([x, phases], dim=1)
        return x

    def forward(self, x):
        """ Forward pass. """
        if self.normalization:
            i = x
            x = self.normalization_layer(x)
            x = torch.clip(x,  -10, 10) # TODO: remove clipping
        if self.phases:
            phases = self.phase_layer(x)
            x = self._concat(x, phases)
        x = self.base_block(x)
        return x
