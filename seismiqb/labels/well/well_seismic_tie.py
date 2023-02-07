""" Mixins and utilities for well-seismic tie.  """

import numpy as np
import torch

from scipy.signal import butter, sosfilt, sosfiltfilt
from torch.nn import functional as F
from numba import njit

from xitorch.interpolate import Interp1D    # Implementation from here: seems to be working fine
                                            # https://github.com/xitorch/xitorch


# Mixin class for `Well` to simplify dt-ticks optimization for well-seismic tie
class OptimizationMixin:
    """ Utilities for optimization of match between seismic data and well logs. """
    @staticmethod
    def get_relevant_seismic_slice(seismic_time, log_time, log_values):
        """ Get slice of seismic time-ticks, where we can linearly interpolate values of a given log.
        That is, all ticks from the resulting slice are bound from left and right with non-nan values of
        a given log.
        """
        nonnan_indices = np.where(np.isfinite(log_values))[0]
        first, last = nonnan_indices[0], nonnan_indices[-1]

        start = np.searchsorted(seismic_time, log_time[first])
        stop = np.searchsorted(seismic_time, log_time[last])
        return slice(start, stop)

    @staticmethod
    def lowpass_filter(curve, order=4, highcut=30, fs=3000, forward_backward=False, pad_width=None):
        """ Lowpass filter of curve. Used to filter out spikey frequencies in well-logs.
        NOTE: fix bugs in `Well.compute_filtered_log` and merge two methods into one.
        """
        sos = butter(order, highcut, btype='lowpass', fs=fs, output='sos')

        # Add pre-padding to remove edge-effect
        if pad_width is not None:
            if isinstance(pad_width, int):
                pad_width = (pad_width, 0)

            curve = np.pad(curve, pad_width, mode='edge')
        else:
            pad_width = (0, 0)
        cut_slice = slice(pad_width[0], len(curve) - pad_width[1])

        if not forward_backward:
            return sosfilt(sos, curve)[cut_slice]
        else:
            return sosfiltfilt(sos, curve)[cut_slice]

    @staticmethod
    def compute_reflectivity_(impedance):
        """ Compute and fetch reflectivity coefficients out of a vector/array/tensor of impedance.
        """
        module = torch if isinstance(impedance, torch.Tensor) else np
        reflectivity = getattr(module, 'zeros_like')(impedance)

        reflectivity[..., 1:] = ((impedance[..., 1:] - impedance[..., :-1]) /
                                 (impedance[..., 1:] + impedance[..., :-1]))
        reflectivity[..., 0:1] = reflectivity[..., 1:2]
        return reflectivity

    @staticmethod
    def resample_log(seismic_time, log_time, log_values):
        """ Take log given in log time and resample it (using linear interpolation) in seismic time.
        """
        if isinstance(seismic_time, torch.Tensor):
            # Resample impedance.
            # Torch implementation of interpolation. Has semantics similar to that of `np.interp`.
            # Supports `backward` and corectly computes gradients.
            log_in_seismic_time = Interp1D(log_time, log_values, method='linear')(seismic_time)
        else:
            log_in_seismic_time = np.interp(x=seismic_time, xp=log_time, fp=log_values)

        return log_in_seismic_time

    @classmethod
    def compute_synthetic_(cls, impedance=None, reflectivity=None, impulse=None):
        """ Compute and fetch synthetic seismic out of a vector of reflectivity, given impulse.
        """
        if impedance is not None:
            reflectivity = cls.compute_reflectivity_(impedance)

        if isinstance(reflectivity, torch.Tensor):
            # Case of incoming torch-tensors.
            if isinstance(impulse, (list, np.ndarray)):
                impulse = torch.tensor(impulse, device=reflectivity.get_device())

            reflectivity = reflectivity.reshape(1, 1, -1)
            impulse = impulse.reshape(1, 1, -1)

            return F.conv1d(reflectivity, impulse, padding='same').reshape(-1)

        # Case of numpy-arrays.
        return np.convolve(reflectivity, impulse, mode='same')

    @staticmethod
    def nancorrelation(array1, array2):
        """ Compute correlation and ignore nan in both `array1` and `array2`.
        """
        array1, array2 = array1 - np.nanmean(array1), array2 - np.nanmean(array2)
        covariation = np.nanmean(array1 * array2)
        std1, std2 = np.sqrt(np.nanmean(array1**2)), np.sqrt(np.nanmean(array2**2))
        return covariation / (std1 * std2)

    @staticmethod
    def torch_correlation(array1, array2):
        """
        """
        array1, array2 = array1 - torch.mean(array1), array2 - torch.mean(array2)
        covariation = torch.mean(array1 * array2)
        std1, std2 = torch.sqrt(torch.mean(array1**2)), torch.sqrt(torch.mean(array2**2))
        return covariation / (std1 * std2)


    @classmethod   # This one can actually be a method. It would take column names of `self.data` as arguments.
    def measure_seismic_tie_quality(cls, seismic_time, well_time, seismic_curve, impedance_log,
                                    impulse, metric='corr'):
        """ Measure the quality of well-seismic tie. Recalculates the synthetic seismic in seismic
        time ticks and compares the synthetic with the original using one of the supported metrics.
        Most commonly used metric is correlation.

        NOTE: convenience function; used for improving the well-seismic tie by optimizing
        (`scipy.optimize`/`torch`-optimization) one of the given arrays, e.g. `well_time`-ticks or
        `impulse`.
        """
        if isinstance(seismic_time, torch.Tensor):
            # Select quality function: the larger the value, the better.
            if metric in ('corr', 'correlation', 'corrcoeff'):
                match_function = cls.torch_correlation
            else:
                raise ValueError(f'Unknown metric {metric} for `numpy`-version!')
        else:
            # Select quality function: the larger the value, the better.
            if metric in ('L2', 'l2'):
                match_function = lambda x, y: -np.nansum(np.square(x - y))
            elif metric in ('mse', 'MSE'):
                match_function = lambda x, y: -np.nanmean(np.square(x - y))
            elif metric in ('corr', 'correlation', 'corrcoeff'):
                match_function = cls.nancorrelation
            else:
                raise ValueError(f'Unknown metric {metric} for `torch`-version of the function!')

        log_in_seismic_time = cls.resample_log(seismic_time, well_time, impedance_log)
        synthetic = cls.compute_synthetic_(impedance=log_in_seismic_time, impulse=impulse)

        result = match_function(synthetic, seismic_curve)
        return result

    @classmethod   # This one can also be a method.
    def optimize_well_time(cls, start_well_time, seismic_time, seismic_curve, impedance_log, impulse,
                           dt_bounds_multipliers=(.95, 1.05), t0_bounds_addition=(-1e-4, 1e-4), n_iters=5000,
                           device='cuda:0', optimizer='Adam', optimizer_kwargs=None):
        """ Improve seismic tie by optimising well-time ticks. For the procedure consider the impulse fixed.
        """
        # Make start point of dt's.
        start_x0_dt = np.diff(start_well_time, prepend=0)

        # Make optimization boundaries for dt's and x0.
        bounds_numpy = [start_x0_dt * dt_bounds_multipliers[0], start_x0_dt * dt_bounds_multipliers[1]]
        bounds_numpy[0][0] = start_x0_dt[0] + t0_bounds_addition[0]
        bounds_numpy[1][0] = start_x0_dt[0] + t0_bounds_addition[1]
        bounds = [torch.from_numpy(data).to(device, dtype=torch.float32) for data in bounds_numpy]

        # Move arrays to needed device.
        impulse, seismic_curve, impedance_log, seismic_time = [
            torch.from_numpy(array).to(device=device, dtype=torch.float32)
            for array in (impulse, seismic_curve, impedance_log, seismic_time)
            ]

        # Init variables of the model using chosen start point.
        variables = torch.tensor(start_x0_dt, device=device, dtype=torch.float32, requires_grad=True)

        # Select and set up the optimizer.
        optimizer = optimizer or 'Adam'
        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)

        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = optimizer((variables, ), **optimizer_kwargs)

        # Run train loop.
        loss_history = []
        for _ in range(n_iters):
            # Reset grads.
            optimizer.zero_grad()

            # NOTE: only well_time needs to be recalculated - these are the time ticks where impedance-values
            # are known. Impedance values and seismic time ticks do not change.
            # NOTE: perhaps implement topK later.
            current_well_time = torch.cumsum(variables, dim=0)
            loss = -cls.measure_seismic_tie_quality(seismic_time, current_well_time, seismic_curve,
                                                    impedance_log, impulse, metric='corr')

            loss.backward()
            loss_history.append(float(loss.detach().cpu().numpy()))

            # Update variables.
            optimizer.step()

            # Apply constraints.
            with torch.no_grad():
                variables.clamp_(bounds[0], bounds[1])

        # Fetch resulting well_time and loss_history.
        final_well_time = np.cumsum(variables.detach().cpu().numpy())
        return final_well_time, loss_history


# Utilities for impulse estimation.
def symmetric_wavelet_estimation(seismic_trace, wavelet_length=60, normalize=True):
    """ Commonly used procedure for wavelet-estimation. Resulting wavelet has length of
    2 * (wavelet_length // 2) and has its peak in the center of the range.
    """
    power_spectrum = np.abs(np.fft.rfft(seismic_trace))

    # Create symmetric wavelet in time.
    wavelet = np.real(np.fft.irfft(power_spectrum)[:wavelet_length // 2])
    wavelet = np.concatenate((np.flipud(wavelet), wavelet), axis=0)
    if normalize:
        wavelet = wavelet / np.max(wavelet)

    return wavelet
@njit
def compute_cross_correlation_1d(signal, lag_size=31):
    """ Get cross-correlation vector of a 1d-trace. The resulting length of the vector is
    2 * lag_size + 1.
    """
    cross = np.zeros(2 * lag_size + 1, dtype=np.float32)
    for i in range(-lag_size, lag_size + 1):
        ctr = i + lag_size
        i = abs(i)
        v1 = signal[i:]
        v2 = signal[:len(v1)]

        cross[ctr] = np.corrcoef(v1, v2)[0, 1]
    return cross

@njit
def compute_cross_correlation_3d(data, lag_size=31):
    """ Vross correlation of 3d-array along the last axis.
    """
    cross = np.zeros(data.shape[:2] + (2*lag_size + 1, ), dtype=np.float32)
    for i in range(cross.shape[0]):
        for j in range(cross.shape[1]):
            cross[i, j] = compute_cross_correlation_1d(data[i, j], lag_size=lag_size)

    return cross

def compute_frequency_amplitides(data):
    """ Calculate Fourier-amplitudes of real data in the last dimension. If the data
    is not 1d, take average over the signals.
    """
    amplitudes = np.abs(np.fft.rfft(data))
    if amplitudes.ndim > 1:
        amplitudes = np.mean(amplitudes, axis=tuple(range(amplitudes.ndim - 1)))

    return amplitudes

def compute_frequency_phases(data):
    """ Calculate Fourier-phases of real data in the last dimension. If the data is not 1d,
    take average over the signals.
    """
    phases = np.angle(np.fft.rfft(data))
    if phases.ndim > 1:
        phases = np.mean(phases, axis=tuple(range(phases.ndim-1)))

    return phases

def construct_wavelet(amplitudes, phases, wavelet_length=None):
    """ Construct a wavelet from vectors of amplitudes and phases given in frequency-space.
    """
    wavelet = np.fft.irfft(amplitudes * np.exp(1j * phases), n=wavelet_length)
    return np.real(wavelet)

def estimate_wavelet_from_crosscor(seismic_data=None, amplitudes=None, phases=None, lag_size=61, normalize=True,
                                   t_shift=None, wavelet_length=121):
    """ Estimate wavelet of length=2 * lag_size [or len(amplitudes)] (+1) from either raw seismic data or precomputed
    amplitudes.
    """
    if amplitudes is None:
        # Get amplitudes spectrum estimation neglecting the phase.
        crosscorr_function = compute_cross_correlation_3d if seismic_data.ndim > 1 else compute_cross_correlation_1d
        cross = crosscorr_function(seismic_data, lag_size=lag_size)

        # Under the assumptions of reflectivity being white noise, the spectrum-power of cross-correlation
        # is the squared spectrum power of the wavelet.
        amplitudes = np.sqrt(compute_frequency_amplitides(cross))

    # Construct the phases of the wavelet if not given.
    # Use `t_shift` to "np.roll" the wavelet and move its center to a needed location.
    if phases is None:
        phases = np.zeros_like(amplitudes) if t_shift is None else t_shift * np.arange(0, len(amplitudes))

    # Assemble the wavelet using the power spectrum and constructed phases.
    wavelet = np.fft.irfft(amplitudes * np.exp(1j * phases), n=wavelet_length)
    if normalize:
        wavelet = wavelet / np.max(wavelet)

    return np.real(wavelet)
