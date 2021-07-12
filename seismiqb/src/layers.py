""" Special layers to incapsulate attribute computation into neural network. """
import torch
import numpy as np
from torch import nn

from ..batchflow.models.torch import ResBlock



class InstantaneousPhaseLayer(nn.Module):
    """ Instantaneous phase comnputation. """
    def __init__(self, inputs=None, continuous=False, enable=True, **kwargs):
        super().__init__()
        self.continuous = continuous
        self.enable = enable

    def _hilbert(self, x):
        """ Hilbert transformation. """
        N = x.shape[-1]
        x = torch.stack([x, torch.zeros_like(x)], dim=-1)
        fft = torch.fft(x, signal_ndim=1, normalized=False)

        h = torch.zeros(N, device=x.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        if x.ndim > 1:
            shape = [1] * x.ndim
            shape[-2] = N
            h = h.view(*shape)

        result = torch.ifft(fft * h, signal_ndim=1)
        return result

    def _angle(self, x):
        """ Compute angle of complex number. """
        res = torch.atan(x[..., 1] / x[..., 0])
        res[x[..., 0] == 0] = np.pi
        res = res % (2 * np.pi) - np.pi
        if self.continuous:
            res = torch.abs(res)
        return res

    def forward(self, x):
        """ Forward pass. """
        if self.enable:
            x = self._hilbert(x)
            x = self._angle(x)
        return x

class InputLayer(nn.Module):
    """ Input layer with possibility of instantaneous phase concatenation. """
    def __init__(self, inputs, phases=False, continuous=False, base_block=ResBlock, **kwargs):
        super().__init__()
        self.phases = phases
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
        if self.phases:
            phases = self.phase_layer(x)
            x = self._concat(x, phases)
        x = self.base_block(x)
        return x
