""" Mixins and utilities for well-seismic tie.  """

import numpy as np
import torch

from scipy.signal import butter, sosfilt, sosfiltfilt
from torch.nn import functional as F

from xitorch.interpolate import Interp1D    # Implementation from here: seems to be working fine
                                            # https://github.com/xitorch/xitorch


# Mixin class for `Well` to simplify dt-ticks optimization for well-seismic tie
class OptimizationMixin:
    """ Utilities for optimization of match between seismic data and well logs. """
    @staticmethod
    def get_relevant_seismic_slice(seismic_time, log_time, log_values):
        """
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
    def nancorrelation(x, y):
        """
        """
        x_, y_ = [np.nanmean(v) for v in (x, y)]
        cov = np.nansum((x - x_) * (y - y_))
        var_x, var_y = np.nansum((x - x_)**2), np.nansum((y - y_)**2)
        return cov / (np.sqrt(var_x) * np.sqrt(var_y))

    @staticmethod
    def compute_reflectivity_(impedance):
        """
        """
        reflectivity = np.zeros_like(impedance)
        reflectivity[..., 1:] = ((impedance[..., 1:] - impedance[..., :-1]) /
                                 (impedance[..., 1:] + impedance[..., :-1]))
        reflectivity[..., 0:1] = reflectivity[..., 1:2]
        return reflectivity

    @staticmethod
    def compute_synthetic_(reflectivity, impulse):
        """
        """
        return np.convolve(reflectivity, impulse, mode='same')

    @staticmethod
    def move_to_device(array, device='cuda:0'):
        """ Move array/torch tensor to CPU/GPU.
        """
        device = device.lower()
        if 'cpu' in device:
            result = array.detach().cpu()
        elif 'np' in device or 'numpy' in device:
            result = array.detach().cpu().numpy()
        else:
            result = torch.tensor(array, device=device, dtype=torch.float32)
        return result

    @staticmethod
    def torch_compute_reflectivity(impedance):
        """
        """
        reflectivity = torch.zeros_like(impedance)
        reflectivity[..., 1:] = ((impedance[..., 1:] - impedance[..., :-1]) /
                                 (impedance[..., 1:] + impedance[..., :-1]))
        reflectivity[..., 0:1] = reflectivity[..., 1:2]
        return reflectivity

    @staticmethod
    def torch_compute_synthetic(reflectivity, impulse):
        """
        """
        if isinstance(impulse, (list, np.ndarray)):
            impulse = torch.tensor(impulse, device=reflectivity.get_device())

        reflectivity = reflectivity.reshape(1, 1, -1)
        impulse = impulse.reshape(1, 1, -1)

        result = F.conv1d(reflectivity, impulse, padding='same')
        return torch.squeeze(result)

    @staticmethod
    def torch_correlation(x, y):
        """
        """
        x_, y_ = [torch.mean(v) for v in (x, y)]
        cov = torch.sum((x - x_) * (y - y_))
        var_x, var_y = torch.sum((x - x_)**2), torch.sum((y - y_)**2)
        return cov / (torch.sqrt(var_x) * torch.sqrt(var_y))


    @classmethod   # This one can actually be a method. It would take column names of `self.data` as arguments.
    def measure_seismic_tie_quality(cls, seismic_time, well_time, seismic_curve, impedance_log,
                                    impulse, metric='corr'):
        """
        """
        log_in_seismic_time = np.interp(x=seismic_time, xp=well_time, fp=impedance_log)

        if metric in ('L2', 'l2'):
            match_function = lambda x, y: np.nansum(np.square(x - y))
        elif metric in ('mse', 'MSE'):
            match_function = lambda x, y: np.nanmean(np.square(x - y))
        elif metric in ('corr', 'correlation', 'corrcoeff'):
            match_function = cls.nancorrelation
        else:
            raise ValueError('Unknown metric!')

        reflectivity = cls.compute_reflectivity_(log_in_seismic_time)
        synthetic = cls.compute_synthetic_(reflectivity, impulse)

        result = match_function(synthetic, seismic_curve)
        return result


    @classmethod
    def torch_measure_seismic_tie_quality(cls, seismic_time, well_time, seismic_curve, impedance_log,
                                          impulse, metric='corr'):
        """
        """
        interpolator = Interp1D(well_time, impedance_log, method='linear')
        log_in_seismic_time = interpolator(seismic_time)

        reflectivity = cls.torch_compute_reflectivity(log_in_seismic_time)
        synthetic = cls.torch_compute_synthetic(reflectivity, impulse=impulse)

        # NOTE: add more options later - perhaps `topK` to fit only important peaks
        if metric in ('corr', 'correlation', 'corrcoeff'):
            match_function = cls.torch_correlation
        else:
            raise ValueError(f'Metric function {metric} is not supported!')

        result = match_function(synthetic, seismic_curve)
        return result


    @classmethod   # This one can also be a method.
    def improve_seismic_tie(cls, start_well_time, seismic_time, seismic_curve, impedance_log, impulse,
                            dt_bounds_multipliers=(.95, 1.05), t0_bounds_addition=(-1e-4, 1e-4), learning_rate=5e-9,
                            n_iters=5000, device='cuda:0'):
        """
        """
        # Make start point of dt's
        start_x0_dt = np.diff(start_well_time, prepend=0)

        # Make optimization boundaries for dt's and x0
        bounds_numpy = [start_x0_dt * dt_bounds_multipliers[0], start_x0_dt * dt_bounds_multipliers[1]]
        bounds_numpy[0][0] = start_x0_dt[0] + t0_bounds_addition[0]
        bounds_numpy[1][0] = start_x0_dt[0] + t0_bounds_addition[1]
        bounds = [cls.move_to_device(data, device) for data in bounds_numpy]

        # Move arrays to needed device
        impulse, seismic_curve, impedance_log, seismic_time = [cls.move_to_device(array, device=device)
                                                               for array in (impulse, seismic_curve,
                                                                             impedance_log, seismic_time)]

        # Init variables of the model using chosen start point
        optimized = torch.tensor(start_x0_dt, device=device, dtype=torch.float32, requires_grad=True)

        # Zeros-vector for applying constraint
        torch_zeros = torch.zeros_like(optimized)

        # Train loop
        loss_history = []
        for _ in range(n_iters):
            # Reset grads
            if optimized.grad is not None:
                optimized.grad.zero_()

            # Note: only well_time needs to be recalculated - these are the time ticks where impedance-values are known
            # Impedance values and seismic time ticks do not change
            current_well_time = torch.cumsum(optimized, dim=0)
            loss = -cls.torch_measure_seismic_tie_quality(seismic_time, current_well_time, seismic_curve,
                                                          impedance_log, impulse, metric='corr')  # NOTE: implement top-k and other options later

            loss.backward()
            loss_history.append(-loss.detach().cpu().numpy())

            # Update grads
            with torch.no_grad():
                optimized.sub_(optimized.grad, alpha=learning_rate)

                # Apply constraints
                optimized.sub_(torch.maximum(optimized - bounds[1], torch_zeros))
                optimized.add_(torch.maximum(bounds[0] - optimized, torch_zeros))

        # Fetch resulting well_time and loss_history
        final_well_time = np.cumsum(cls.move_to_device(optimized, device='numpy'))
        result = (final_well_time, loss_history)
        return result


# Utilities for impulse estimation.
def symmetric_wavelet_estimation(seismic_trace, half_length=31, norm=True):
    """ Commonly used procedure for wavelet-estimation. Resulting wavelet has length of
    2 * half_length + 1.
    """
    power_spectrum = np.abs(np.fft.fft(seismic_trace))

    # Create symmetric wavelet in time.
    wavelet = np.real(np.fft.ifft(power_spectrum)[:half_length])
    wavelet = np.concatenate((np.flipud(wavelet[1:]), wavelet), axis=0)
    if norm:
        wavelet = wavelet / np.max(wavelet)

    return wavelet
