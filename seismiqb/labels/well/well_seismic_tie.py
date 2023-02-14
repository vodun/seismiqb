""" Mixins and utilities for well-seismic tie.  """

import numpy as np
import torch

from scipy.optimize import minimize
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks
from torch.nn import functional as F

from xitorch.interpolate import Interp1D    # Implementation from here: seems to be working fine
                                            # https://github.com/xitorch/xitorch

from ...plotters import plot

# Mixin class for `Well` to simplify dt-ticks optimization for well-seismic tie
class WellMatcher:
    """ Utilities for optimization of match between seismic data and well logs. """
    def __init__(self, seismic_time, well_time, seismic_curve, impedance_log):
        """ Store the state of the matcher.
        """
        self.seismic_time = seismic_time
        self.well_time = well_time
        self.seismic_curve = seismic_curve
        self.impedance_log = impedance_log

        self.tie_crosscorrelation = None


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
    def compute_synthetic_(cls, impedance=None, reflectivity=None, wavelet=None):
        """ Compute and fetch synthetic seismic out of a vector of reflectivity, given wavelet.
        """
        if impedance is not None:
            reflectivity = cls.compute_reflectivity_(impedance)

        if isinstance(reflectivity, torch.Tensor):
            # Case of incoming torch-tensors.
            if isinstance(wavelet, (list, np.ndarray)):
                wavelet = torch.tensor(wavelet, device=reflectivity.get_device())

            reflectivity = reflectivity.reshape(1, 1, -1)
            wavelet = wavelet.reshape(1, 1, -1)

            return F.conv1d(reflectivity, wavelet, padding='same').reshape(-1)

        # Case of numpy-arrays.
        return np.convolve(reflectivity, wavelet, mode='same')

    @classmethod
    def resample_and_compute_synthetic(cls, seismic_time, well_time, impedance_log, wavelet):
        """ """
        resampled_impedance = cls.resample_log(seismic_time, well_time, impedance_log)
        recreated = cls.compute_synthetic_(impedance=resampled_impedance, wavelet=wavelet)
        return recreated

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


    @classmethod
    def measure_seismic_tie_quality(cls, seismic_time, well_time, seismic_curve, impedance_log,
                                    wavelet, metric='corr'):
        """ Measure the quality of well-seismic tie. Recalculates the synthetic seismic in seismic
        time ticks and compares the synthetic with the original using one of the supported metrics.
        Most commonly used metric is correlation.

        NOTE: convenience function; used for improving the well-seismic tie by optimizing
        (`scipy.optimize`/`torch`-optimization) one of the given arrays, e.g. `well_time`-ticks or
        `wavelet`.
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
        synthetic = cls.compute_synthetic_(impedance=log_in_seismic_time, wavelet=wavelet)

        result = match_function(synthetic, seismic_curve)
        return result

    def apply(self, src, dst, transform):
        """
        """
        setattr(self, dst, transform(getattr(self, src)))

    def optimize_well_time(self, src='well_time', dst='well_time', dst_history=None, seismic_time_slice=None,
                           dt_bounds_multipliers=(.95, 1.05), t0_bounds_addition=(-1e-4, 1e-4), n_iters=5000,
                           device='cuda:0', optimizer='Adam', optimizer_kwargs=None, flip_wavelet=True):
        """ Improve seismic tie by optimising well-time ticks. For the procedure consider the wavelet fixed.
        """
        start_well_time, seismic_time, seismic_curve, impedance_log, wavelet = [
            getattr(self, name) for name in (src, 'seismic_time', 'seismic_curve', 'impedance_log', 'wavelet')
            ]

        if dst_history is not None:
            setattr(self, dst_history, start_well_time)

        if flip_wavelet:
            wavelet = wavelet[::-1].copy()

        # Cut needed time slice if requested
        seismic_time_slice = seismic_time_slice or slice(None, None)
        seismic_time, seismic_curve = seismic_time[seismic_time_slice], seismic_curve[seismic_time_slice]

        # Make start point of dt's.
        start_x0_dt = np.diff(start_well_time, prepend=0)

        # Make optimization boundaries for dt's and x0.
        bounds_numpy = [start_x0_dt * dt_bounds_multipliers[0], start_x0_dt * dt_bounds_multipliers[1]]
        bounds_numpy[0][0] = start_x0_dt[0] + t0_bounds_addition[0]
        bounds_numpy[1][0] = start_x0_dt[0] + t0_bounds_addition[1]
        bounds = [torch.from_numpy(data).to(device, dtype=torch.float32) for data in bounds_numpy]

        # Move arrays to needed device.
        wavelet, seismic_curve, impedance_log, seismic_time = [
            torch.from_numpy(array).to(device=device, dtype=torch.float32)
            for array in (wavelet, seismic_curve, impedance_log, seismic_time)
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
            loss = -self.measure_seismic_tie_quality(seismic_time, current_well_time, seismic_curve,
                                                     impedance_log, wavelet, metric='corr')

            loss.backward()
            loss_history.append(float(loss.detach().cpu().numpy()))

            # Update variables.
            optimizer.step()

            # Apply constraints.
            with torch.no_grad():
                variables.clamp_(bounds[0], bounds[1])

        # Fetch resulting well_time and loss_history.
        final_well_time = np.cumsum(variables.detach().cpu().numpy())

        setattr(self, dst, final_well_time)

        return final_well_time, loss_history

    def optimize_wavelet(self, src='wavelet', dst='wavelet', dst_history=None, cut_frequency=8, delta=.9, **kwargs):
        """
        """
        # Compute reflectivity given current well time
        impedance = self.resample_log(self.seismic_time, self.well_time, self.impedance_log)
        reflectivity = self.compute_reflectivity_(impedance)

        # Functional for wavelet optimization
        start_wavelet = getattr(self, src)
        if dst_history is not None:
            setattr(self, dst_history, start_wavelet)

        functional = WaveletOptimizationFactory(start_wavelet, reflectivity, self.seismic_curve, cut_frequency, delta)

        # Perform minimization
        optimization_results = minimize(functional, functional.get_x0, bounds=functional.get_bounds, **kwargs)
        wavelet = functional.compute_wavelet(optimization_results['x'])

        setattr(self, dst, wavelet)

        return wavelet

    def compute_tie_crosscorrelation(self, n_samples=10000, limits=(-.5, 1.5), compute_peaks=True, dinstance_peaks=10):
        """ Compute tie crosscorrelation function of comparing recorded seismic and the synthetic seismic,
        generated from logs. Allows to determine the shift to apply to `well_time`.
        """
        seismic_time, well_time, seismic_curve, impedance_log, wavelet = [
            getattr(self, src) for src in ('seismic_time', 'well_time', 'seismic_curve', 'impedance_log', 'wavelet')
            ]

        # Compute crosscorrelation-values on a grid of points.
        shifts = np.linspace(*limits, n_samples)
        values = [self.measure_seismic_tie_quality(seismic_time, well_time + shift, seismic_curve, impedance_log,
                                                   wavelet, metric='corr') for shift in shifts]

        # Determine peaks if needed.
        if compute_peaks:
            peak_indices = find_peaks(values, distance=dinstance_peaks)[0]

            # Sort the array of peak indices to start with those indices that have
            # the largest corresponding value of match.
            peak_indices = peak_indices[np.argsort([values[index] for index in peak_indices])[::-1]]
        else:
            peak_indices = []

        # Store the computed results.
        self.tie_crosscorrelation = {'shifts': shifts, 'values': values, 'peak_indices': peak_indices,
                                     'peak_shifts': [shifts[ix] for ix in peak_indices],
                                     'peak_values': [values[ix] for ix in peak_indices]}

    # Visualization
    def show_tie_crocccorrelation(self, n_samples=10000, limits=(-.5, 1.5), n_peaks=3, **kwargs):
        """ Compute and show crosscorrelation function of recorded seismic and the synthetic one, generated from logs.
        Allows to see whether we need to apply a shift to well_time.
        """
        if self.tie_crosscorrelation is None:
            self.compute_tie_crosscorrelation(n_samples=n_samples, limits=limits, compute_peaks=n_peaks > 0)

        shifts, values, peak_indices = [self.tie_crosscorrelation[key] for key in ('shifts', 'values', 'peak_indices')]

        # Default plot parameters
        defaults = {'xlabel': 'SHIFT, SECONDS', 'ylabel': 'MATCH QUALITY',
                    'xlabel_fontsize': 22, 'ylabel_fontsize': 22, 'figsize': (14, 4),
                    'title': 'CROSS-CORRELATION RECREATED VS RECORDED SEISMIC'}

        # Update defaults and plot
        kwargs = {'mode': 'curve', **defaults, **kwargs}
        plotter = plot((shifts, values), **kwargs)

        # Show peaks if needed
        if n_peaks > 0:
            axis = plotter.subplots[0].ax

            # Add vertical lines in locations of peaks.
            for index in peak_indices[:n_peaks]:
                axis.axvline(shifts[index], linestyle='dashed', color='orange', alpha=.6,
                             label=f'SHIFT: {shifts[index]:.2f}\nCORR: {values[index]:.2f}')

            axis.legend()

        return plotter

    def show_ranges(self, **kwargs):
        """ Show ranges of seismic time and well time on one plot. Needed to select limits for computing
        crosscorrelation and selecting the global shift.
        """
        seismic_range = np.nanmin(self.seismic_time), np.nanmax(self.seismic_time)
        well_range = np.nanmin(self.well_time), np.nanmax(self.well_time)

        data = [(seismic_range, (1, 1)), ((well_range), (2, 2))]

        # Default plot parameters
        defaults = {'label': ['SEISMIC TIME RANGE', f'WELL TIME RANGE'], 'curve_linewidth': [4, 4],
                    'curve_marker': 'o', 'curve_markersize': 12,
                    'xlabel': 'TIME, SECONDS', 'ylabel': 'SEISMIC/WELL',
                    'xlabel_fontsize': 22, 'ylabel_fontsize': 22,
                    'title': 'WELL TIME RANGE WITH BEST SHIFT VS SEISMIC TIME RANGE'}

        # Update defaults and plot
        kwargs = {'mode': 'curve', **defaults, **kwargs}
        plotter = plot(data, **kwargs)

        return plotter

    def show_tie_comparison(self, src_well_time=('well_time_initial', 'well_time'), src_wavelet=('wavelet', 'wavelet'),
                            seismic_time_slice=None, synthetic_postfix=('BEFORE TIE', 'AFTER TIE'),  **kwargs):
        """ Visually compare several (1 or 2 usually) ties: show recorded seismic along with recreated synthetics
        corresponding to the ties. Include computed correlation; use chosen slice of seismic time for demonstration
        and computation of correlation.
        """
        if isinstance(src_well_time, str):
            src_well_time = (src_well_time, )
        if isinstance(src_wavelet, str):
            src_wavelet = (src_wavelet, )
        if isinstance(synthetic_postfix, str):
            synthetic_postfix = (synthetic_postfix, )

        # Take cut out of seismic curves if needed
        seismic_time_slice = seismic_time_slice or slice(None, None)
        seismic_time = self.seismic_time[seismic_time_slice]
        seismic_curve = self.seismic_curve[seismic_time_slice]

        # Recreate synthetics and compute correlation for each one of them
        recreated = [
            self.resample_and_compute_synthetic(seismic_time, getattr(self, src_well_time_),
                                                self.impedance_log, getattr(self, src_wavelet_))
                                                for src_well_time_, src_wavelet_ in zip(src_well_time, src_wavelet)
            ]
        correlation = [self.nancorrelation(recreated_, seismic_curve) for recreated_ in recreated]

        # Default plot parameters
        defaults = {'title': f'SYNTHETIC VS RECREATED SEISMIC',
                    'label': ['SEISMIC'] + [f'SYNTHETIC {postfix}, CORR: {correlation_: .3f}'
                                            for correlation_, postfix in
                                            zip(correlation, synthetic_postfix)],
                    'figsize': (23, 5), 'curve_alpha': [.8, 1, 1],
                    'curve_linestyle': ['solid', 'dashed', 'solid'],
                    'curve_linewidth': [2.5, 2, 2],
                    'curve_color': ['sandybrown', 'lightpink', 'cornflowerblue']}

        # Update defaults and plot
        kwargs = {'mode': 'curve', **defaults, **kwargs}
        plotter = plot([seismic_curve] + recreated, **kwargs)

        return plotter

# Utilities for wavelet estimation.
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


class WaveletOptimizationFactory:
    """ One can use the instances of this class to optimize wavelet. The default version
    fixes amplitudes and allows to optimize phases for most important frequencies, starting
    from given position.
    """
    def __init__(self, start_wavelet, reflectivity, seismic_curve, cut_frequency=8, delta=.9):
        """ Store variables that we'll need to run the functional.
        """
        self.length = len(start_wavelet)
        self.amplitudes = compute_frequency_amplitides(start_wavelet)
        self.phases = compute_frequency_phases(start_wavelet)
        self.cut_frequency = cut_frequency
        self.seismic_curve = seismic_curve
        self.delta = delta
        self.reflectivity = reflectivity

    def compute_wavelet(self, x):
        phases = np.copy(self.phases)
        phases[:self.cut_frequency] = x

        wavelet = construct_wavelet(self.amplitudes, phases, wavelet_length=self.length)
        return wavelet

    def __call__(self, x):
        wavelet = self.compute_wavelet(x)
        synthetic = OptimizationMixin.compute_synthetic_(reflectivity=self.reflectivity, wavelet=wavelet)

        return -OptimizationMixin.nancorrelation(self.seismic_curve, synthetic)

    @property
    def get_x0(self):
        """ Get phases for the most important frequencies. That is, starting point for phases
        optimization.
        """
        return self.phases[:self.cut_frequency]

    @property
    def get_bounds(self):
        """ Get phases for the most important frequencies. That is, starting point for phases
        optimization.
        """
        return [(self.phases[i] - self.delta, self.phases[i] + self.delta) for i in range(self.cut_frequency)]


def show_wavelet(wavelet, cut_frequency=8, **kwargs):
    """ Show wavelet along with its most important properties. Demonstrates 4 subplots in total:
    1) the wavelet itself 2) its amplitudes 3) its phases 4) recreated wavelet from `cut_frequency`
    amplitudes.
    """
    amplitudes, phases = compute_frequency_amplitides(wavelet), compute_frequency_phases(wavelet)
    cut_frequency = 8

    # Construct wavelet from the start of its spectrum.
    amplitudes_ = amplitudes.copy()
    amplitudes_[cut_frequency:] = 0
    restored = construct_wavelet(amplitudes_, phases, wavelet_length=len(wavelet))
    data = [wavelet, amplitudes, phases, [wavelet, restored]]

    defaults = {'label': ['', '', '', [f'RESTORED FROM {cut_frequency} FREQUENCIES', 'FULL']],
                'xlabel': ['', 'FREQUENCY', 'FREQUENCY', 'TIME'],
                'ylabel': ['', 'AMPLITUDE', 'PHASE', ''],
                'title': ['WAVELET/TIME', 'AMPLITUDES/FREQUENCIES', 'PHASES/FREQUENCIES',
                          'RESTORED WAVELET FROM FULL/CUT SPECTRUM'],
                'curve_alpha': [1, 1, 1, [1, 1]], 'curve_linewidth': [2, 2, 2, [3, 2]], 'nrows': 2, 'ncols': 2,
                'xlabel_fontsize': 18, 'ylabel_fontsize': 18,
                'curve_linestyle': ['-', '-', '-', ['-', '--']]}

    kwargs = {'mode': 'curve', **defaults, **kwargs}

    plotter = plot(data, **kwargs)

    return plotter
