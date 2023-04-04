""" !!. """
import numpy as np
import pandas as pd

from scipy.signal import ricker, find_peaks
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from sklearn.linear_model import Ridge

from ..geometry import array_to_segy
from ..plotters import plot




class WellSeismicMatcher:
    """ !!. """
    # add altitude
    def __init__(self, well, field, coordinates):
        # Well data
        self.well = well
        self.well_bounds = None
        self.well_times = None
        self.well_impedance = None
        self.well_reflectivity = None

        # Seismic data: extract trace at well location
        self.field = field
        self.seismic_times = np.arange(0, field.depth, dtype=np.float32) * field.sample_interval * 1e-3 # seconds
        self.seismic_trace = None
        self.coordinates = None

        if coordinates is not None:
            if isinstance(coordinates, dict):
                coordinates = coordinates[self.well.name]
            self.extract_seismic_trace(coordinates)

        self.states = []


    def extract_seismic_trace(self, coordinates):
        """ !!. """
        # TODO: add averaging (copy from 2d matcher), add inclinometry information
        trace = self.field.geometry[coordinates[0], coordinates[1], :]
        self.seismic_trace = trace
        self.coordinates = coordinates


    # Extended initialization
    def process_well(self, impedance_log=None, reflectivity_name='R', recompute_ai=False, recompute_rhob=False,
                     filter_ai=False, filter_dt=False, filter_rhob=False):
        """ !!. """
        #pylint: disable=protected-access
        # Add Gardner's equation to compute density log
        fs = self.well.compute_sampling_frequency()

        # Sonic log, optionally filtered
        dt_values = self.well.DT.values                                                                  # us/ft

        if filter_dt:
            filtration_parameters = {
                'order': 4, 'frequency': 30, 'btype': 'lowpass', 'fs': fs,
                **(filter_dt if isinstance(filter_dt, dict) else {}),
            }
            dt_values = self.well._compute_filtered_log(dt_values, **filtration_parameters)
            self.well['DT_FILTERED'] = dt_values

        # Prepare impedance log
        if impedance_log is not None:
            pass
        elif 'AI' in self.well.keys and not recompute_ai:
            impedance_log = 'AI'
        else:
            if recompute_rhob or 'RHOB' not in self.well.keys:
                # Gardner's equation
                vp = (0.3048 / dt_values) * 1e6                                                          # m/s
                self.well['RHOB_RECOMPUTED'] = 310 * (vp ** 0.25)                                        # g/cm3
                rhob_log = 'RHOB_RECOMPUTED'
            else:
                rhob_log = 'RHOB'

            # Density values, optionally filtered
            rhob_values = self.well[rhob_log].values
            if filter_rhob:
                filtration_parameters = {
                    'order': 4, 'frequency': 30, 'btype': 'lowpass', 'fs': fs,
                    **(filter_rhob if isinstance(filter_rhob, dict) else {}),
                }
                rhob_values = self.well._compute_filtered_log(rhob_values, **filtration_parameters)

            # Recomputed AI. Can omit unit conversions as they dont influence the reflectivity
            self.well['AI_RECOMPUTED'] = rhob_values / (dt_values * 1e-6 / 0.3048)                       # kPa.s/m
            impedance_log = 'AI_RECOMPUTED'

        # Filter impedance log
        if filter_ai:
            filtration_parameters = {
                'order': 4, 'frequency': 30, 'btype': 'lowpass', 'fs': fs,
                **(filter_ai if isinstance(filter_ai, dict) else {}),
            }
            self.well['AI_FILTERED'] = self.well._compute_filtered_log(self.well[impedance_log],
                                                                       **filtration_parameters)
            impedance_log = 'AI_FILTERED'

        self.well.compute_reflectivity(impedance_log=impedance_log, name=reflectivity_name)

        bounds = self.well.get_bounds(dt_values)
        self.well_bounds = slice(bounds[0]-1, bounds[1])
        self.well_times = np.cumsum(dt_values[self.well_bounds]) * 1e-6                                  # seconds
        self.well_impedance = self.well[impedance_log].values[self.well_bounds]
        self.well_reflectivity = self.well[reflectivity_name].values[self.well_bounds]


    def extract_wavelet(self, method='statistical', normalize=False, taper=True, wavelet_length=61, state=-1, **kwargs):
        """ !!. """
        # Prepare trace and (optionally) reflectivity
        trace = self.seismic_trace.copy()
        if taper:
            trace *= np.blackman(len(trace)) # TODO: taper selection, maybe?
        lenhalf, wlenhalf, wlenflag = len(trace) // 2, wavelet_length // 2, wavelet_length % 2

        if method in {'lstsq', 'division'}:
            state = state if isinstance(state, dict) else self.states[state]
            reflectivity = self.resample_to_seismic(seismic_times=self.seismic_times,
                                                    well_times=state['well_times'],
                                                    well_data=np.nan_to_num(self.well_reflectivity))
            if taper:
                reflectivity *= np.blackman(len(trace))

        # Create wavelet
        if method == 'deterministic':
            ...
        elif method == 'ricker':
            # Given `frequency`, one can compute width: a = geometry.sample_rate / (frequency * np.sqrt(2) * np.pi)
            kwargs = {'points': wavelet_length, 'a': 4.5, **kwargs}
            wavelet = ricker(**kwargs)

        elif method == 'ricker_f':
            # ...or just use this method for wavelet creation
            kwargs = {'f': 25, **kwargs}
            f = kwargs['f']
            t = np.arange(wavelet_length) * self.field.sample_interval * 1e-3
            t -= t[wlenhalf]
            pft2 = (np.pi * f * t) ** 2
            wavelet = (1 - 2 * pft2) * np.exp(-pft2)

        elif method == 'lstsq':
            # Fit wavelet coeffs to the current reflectivity / seismic trace
            projection = np.zeros((reflectivity.size, reflectivity.size))
            projection[:wavelet_length, :wavelet_length] = np.eye(wavelet_length)

            reflectivity_toeplitz = toeplitz(reflectivity)
            operator  = np.dot(reflectivity_toeplitz, projection)

            # wavelet = np.linalg.lstsq(op, trace)[0]
            model = Ridge(alpha=0.5, fit_intercept=False)
            model.fit(operator, trace)
            wavelet = model.coef_[:wlenhalf + wlenflag]
            wavelet = np.concatenate((wavelet[::-1], wavelet[wlenflag:]), axis=0)

        else:
            # Compute power spectrum by different algorithms
            if method in {'stats1', 'statistical'}:
                power_spectrum = np.abs(np.fft.rfft(trace))

            elif method in {'stats2', 'autocorrelation'}:
                autocorrelation = np.correlate(trace, trace, mode='same')
                autocorrelation = autocorrelation[lenhalf - wlenhalf : lenhalf + wlenhalf + wlenflag]
                power_spectrum = np.sqrt(np.abs(np.fft.rfft(autocorrelation)))

            elif method in {'division'}:
                power_spectrum = np.fft.rfft(trace) / np.fft.rfft(reflectivity)

            wavelet = np.real(np.fft.irfft(power_spectrum)[:wlenhalf + wlenflag])
            wavelet = np.concatenate((wavelet[::-1], wavelet[wlenflag:]), axis=0)

        if normalize:
            wavelet /= wavelet.max()
        return wavelet


    def init_state(self, wavelet=None, state=-1):
        """ !!. """
        if isinstance(state, dict):
            previous_state = state
        elif self.states:
            previous_state = self.states[state]
        else:
            previous_state = {
                'well_times': self.well_times,
            }

        state = {
            'type': 'init',
            'well_times': previous_state['well_times'],
            'wavelet': wavelet,
        }
        state['correlation'] = self.compute_metric(**state)
        self.states.append(state)


    def change_wavelet(self, keep_bounds=True,
                       method='statistical', normalize=False, taper=True, wavelet_length=61, state=-1, **kwargs):
        """ !!. """
        previous_state = state if isinstance(state, dict) else self.states[state]
        wavelet = self.extract_wavelet(method=method, normalize=normalize, taper=taper,
                                       wavelet_length=wavelet_length, state=state, **kwargs)
        state = {
            'type': 'change_wavelet',
            'well_times': previous_state['well_times'],
            'wavelet': wavelet
        }
        if keep_bounds and 'bounds' in previous_state:
            state['bounds'] = previous_state['bounds']
        state['correlation'] = self.compute_metric(**state)
        self.states.append(state)


    # Helper functions
    def compute_resampled_synthetic(self, well_times=None, wavelet=None, limits=None, multiply=False, **kwargs):
        """ !!. """
        #pylint: disable=protected-access
        _ = kwargs
        limits = limits if limits is not None else slice(None)

        # impedance_resampled = self.resample_to_seismic(seismic_times=self.seismic_times[limits],
        #                                                well_times=well_times,
        #                                                well_data=self.well_impedance)
        # reflectivity_resampled = self.well._compute_reflectivity(impedance_resampled)
        # synthetic_trace = self.well._compute_synthetic(reflectivity_resampled, wavelet=wavelet)
        # return synthetic_trace

        reflectivity_resampled = self.resample_to_seismic(seismic_times=self.seismic_times[limits],
                                                          well_times=well_times,
                                                          well_data=self.well_reflectivity)
        synthetic_trace = self.well._compute_synthetic(reflectivity_resampled, wavelet=wavelet)

        if multiply:
            synthetic_trace *= self.compute_multiplier(self.seismic_trace[limits], synthetic_trace)
        return synthetic_trace

    @staticmethod
    def resample_to_seismic(seismic_times, well_times, well_data):
        """ !!. """
        return np.interp(x=seismic_times, xp=well_times, fp=well_data)
        # return interp1d(x=well_times, y=well_data, kind='slinear',
        #                 bounds_error=False, fill_value=(well_data[0], well_data[-1]))(seismic_times)

    @staticmethod
    def compute_multiplier(seismic_trace, synthetic_trace):
        """ !!. """
        return np.abs(seismic_trace).mean() / np.abs(synthetic_trace).mean()

    def compute_metric(self, metric='correlation', synthetic_trace=None,
                       well_times=None, wavelet=None, limits=None, **kwargs):
        """ !!. """
        _ = kwargs
        limits = limits if limits is not None else slice(None)

        if synthetic_trace is None:
            synthetic_trace = self.compute_resampled_synthetic(well_times=well_times, wavelet=wavelet, limits=limits)

        if metric == 'correlation':
            value = self.correlation(self.seismic_trace[limits], synthetic_trace)
        return value

    @staticmethod
    def correlation(array_0, array_1):
        """ !!. """
        return ((array_0 - array_0.mean()) * (array_1 - array_1.mean())).mean() / (array_0.std() * array_1.std())


    # t0 optimization
    def compute_t0(self, ranges=(-0.5, +1.5), n=1000, limits=None, state=-1, index=0):
        """ !!. """
        previous_state = state if isinstance(state, dict) else self.states[state]
        limits = limits if limits is not None else slice(None)
        well_times = previous_state['well_times']
        wavelet = previous_state['wavelet']

        # Compute correlation values. TODO: can be massively speed up by vectorization (the same as in 2d matching)
        shifts = np.linspace(*ranges, n)
        values = [self.compute_metric(well_times=well_times+shift, wavelet=wavelet, limits=limits)
                  for shift in shifts]
        values = np.array(values)

        # Compute peaks and their corresponding metric values
        peak_indices = find_peaks(values, distance=10, prominence=0.1)[0]
        peak_values = values[peak_indices]

        argsort = np.argsort(peak_values)[::-1]
        peak_indices = peak_indices[argsort]
        peak_shifts = shifts[peak_indices]
        peak_values = values[peak_indices]

        # Select the best t0
        if index == -1:
            t0 = 2 * self.well.index[0] * np.diff(self.well_times)[0] / 0.3048
        else:
            t0 = peak_shifts[index]

        new_well_times = self.well_times + t0
        correlation = self.compute_metric(well_times=new_well_times, wavelet=wavelet)

        # Save state
        state = {
            'type': 'compute_t0',
            'well_times': new_well_times,
            'wavelet': wavelet,
            't0': t0, 'correlation': correlation,
            'shifts': shifts, 'values': values,
            'peak_shifts': peak_shifts, 'peak_values': peak_values,
        }
        self.states.append(state)


    def optimize_t0(self, state=-1, limits=None, **kwargs):
        """ !!. """
        previous_state = state if isinstance(state, dict) else self.states[state]
        limits = limits if limits is not None else slice(None)
        wavelet = previous_state['wavelet']

        if 't0' in previous_state:
            t0_start = previous_state['t0']
        else:
            t0_start = previous_state['well_times'][0] - self.well_times[0]

        def minimization_proxy(x):
            return -self.compute_metric(well_times=self.well_times+x, wavelet=wavelet, limits=limits)

        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-3, 'eps': 1e-6},
            **kwargs
        }
        optimization_results = minimize(minimization_proxy, x0=t0_start, **kwargs)
        t0 = optimization_results['x']

        # Save state
        state = {
            'type': 'optimize_t0',
            'well_times': self.well_times + t0,
            'wavelet': previous_state['wavelet'],
            't0': t0, 'correlation': -optimization_results['fun'],
            'optimization_results': optimization_results,
        }
        self.states.append(state)


    # Extrema optimization
    @staticmethod
    def stretch_well_times(well_times, position, alpha, left_bound, right_bound, **kwargs):
        """ !!.

        |           `left_bound`             `position`              `right_bound`
        |----------------|------------------------o------------------------|-------------------|
        |                       <segment x>              <segment y>

        """
        _ = kwargs

        # Maybe, add taper (~blackman)?
        dt = np.diff(well_times, prepend=0)
        x = dt[left_bound:position]
        y = dt[position:right_bound]

        beta = 1 + (1 - alpha) * x.sum() / y.sum()

        new_dt = dt.copy()
        new_dt[left_bound:position] *= alpha
        new_dt[position:right_bound] *= beta
        return np.cumsum(new_dt)

    def optimize_extrema(self, topk=20, threshold_max=0.050, threshold_min=0.001, threshold_nearest=0.010,
                         threshold_iv_max=500, alpha_bounds=(0.9, 1.1), state=-1, **kwargs):
        """ !!. """
        # `threshold_max` is max amount of SECONDS to shift the extrema position
        # `threshold_min` is min amount of SECONDS to shift the extrema position
        # `threshold_nearest` is minimum distance to already-shifted extrema in SECONDS
        # `threshold_iv_max` is the maximum IV difference in m/s

        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-3, 'eps': 1e-6},
            **kwargs
        }

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times, wavelet = previous_state['well_times'], previous_state['wavelet']
        bounds = previous_state.get('bounds', [1, len(well_times) - 1])

        # Compute current state of synthetic; find extremas on it # TODO: maybe, use peaks of envelope?
        synthetic_trace = self.compute_resampled_synthetic(**previous_state)
        dt = np.diff(well_times, prepend=0)

        values = np.abs(synthetic_trace)
        peak_indices = find_peaks(values, distance=5)[0]
        peak_indices = peak_indices[np.argsort(values[peak_indices])[::-1]] # sort peaks by extrema value

        # For each extrema, check the potential correlation gain by stretching left/right side of it
        results = []
        for index in range(topk):
            #pylint: disable=cell-var-from-loop
            # Locate extreme in well times
            peak_index = peak_indices[index]
            peak_time = self.seismic_times[peak_index]
            peak_position = np.searchsorted(well_times, peak_time)

            # Select left/right segments. Potentially early-stop
            bounds_idx = np.searchsorted(bounds, peak_position)

            if peak_position <= bounds[0] or peak_position >= bounds[-1]:
                continue
            left_bound, right_bound = bounds[bounds_idx-1], bounds[bounds_idx]

            if min(abs(well_times[peak_position] - well_times[left_bound]),
                   abs(well_times[peak_position] - well_times[right_bound])) <= threshold_nearest:
                continue

            x = dt[left_bound:peak_position]
            y = dt[peak_position:right_bound]
            xsum , ysum = x.sum(), y.sum()

            # Optimize via adjusting left-stretches `alpha`
            def minimization_proxy(alpha):
                beta = 1 + (1 - alpha) * xsum / ysum

                new_dt = dt.copy()
                new_dt[left_bound:peak_position] *= alpha
                new_dt[peak_position:right_bound] *= beta

                new_well_times = np.cumsum(new_dt)
                # TODO: can add limits=(left bounds, right bounds)
                return -self.compute_metric(well_times=new_well_times, wavelet=wavelet)

            # Prepare bounds and early stop, if too restrictive
            s = threshold_max / xsum
            tmax = min(x.min() * threshold_iv_max / 0.3048, 0.5)
            optimization_bounds = (max(1 - s, alpha_bounds[0]),
                                   min(1 + s, 1 / (1 - tmax), alpha_bounds[1]))
            if optimization_bounds[0] > optimization_bounds[1]:
                continue

            # Actual optimization
            optimization_results = minimize(minimization_proxy, x0=1., bounds=[optimization_bounds], **kwargs)

            # Check if the stretch on either side is too small / too big
            alpha = optimization_results['x'].item()
            beta = 1 + (1 - alpha) * xsum / ysum
            if (1 - threshold_min / xsum) <= alpha <= (1 + threshold_min / ysum) or \
               (1 - threshold_min / xsum) <= beta  <= (1 + threshold_min / ysum) or \
                beta <= alpha_bounds[0] or beta >= alpha_bounds[1]:
                continue

            iv_diffs_x = 0.3048 * (alpha - 1) / (alpha * x)
            iv_diffs_y = 0.3048 * ( beta - 1) / ( beta * y)
            if np.abs(iv_diffs_x).max() > threshold_iv_max or np.abs(iv_diffs_y).max() > threshold_iv_max:
                continue

            results.append({
                'position': peak_position,
                'alpha': alpha, 'beta': beta,
                'left_bound': left_bound, 'right_bound': right_bound,
                'correlation': -optimization_results['fun'],
                'xsum': xsum, 'ysum': ysum,
                'optimization_bounds': optimization_bounds,
                'optimization_results': optimization_results,
            })

        if len(results) == 0:
            return False

        # Select the best extrema to stretch about
        metrics = [item['correlation'] for item in results]
        index = np.argmax(metrics)
        state = results[index]
        position = state['position']
        new_well_times = self.stretch_well_times(well_times, **state)
        time_shift = well_times[position] - new_well_times[position]

        state.update({
            'type': 'optimize_extrema',
            'well_times': new_well_times,
            'wavelet': wavelet,
            'time_shift': time_shift,
            'time_before': well_times[position], 'time_after': new_well_times[position],
            'correlation_delta': state['correlation'] - previous_state['correlation'],
            'bounds': sorted(bounds + [position]),
        })
        self.states.append(state)
        return True

    def optimize_extremas(self, steps=20, threshold_delta=0.01, verbose=True,
                          topk=20, threshold_max=0.050, threshold_min=0.001, threshold_nearest=0.010,
                          threshold_iv_max=500, alpha_bounds=(0.9, 1.1), **kwargs):
        """ !!. """
        #
        for i in range(steps):
            success = self.optimize_extrema(topk=topk, threshold_max=threshold_max, threshold_min=threshold_min,
                                            threshold_nearest=threshold_nearest,
                                            alpha_bounds=alpha_bounds, threshold_iv_max=threshold_iv_max, **kwargs)
            if not success:
                if verbose:
                    print('Early break: no good adjustment found!')
                break

            state = self.states[-1]
            correlation_delta = state['correlation_delta']

            if verbose:
                correlation = self.compute_metric(**state)
                time_shift = state['time_shift'] * 1000
                alpha, beta = state['alpha'], state['beta']
                print(f'{i:3} :: {correlation=:3.5f} :: {correlation_delta=:3.5f} :: {time_shift=:>+7.4} ms'
                      f'      ||      {alpha=:3.3f} :: {beta=:3.3f}')

            if correlation_delta < threshold_delta:
                if verbose:
                    print('Early break: correlation is no longer increasing!')
                break


    # Pytorch well times optimization
    @staticmethod
    def compute_resampled_synthetic_torch(well_times, well_reflectivity, seismic_times, wavelet):
        """ !!. """
        #pylint: disable=import-outside-toplevel
        import torch
        from xitorch.interpolate import Interp1D
        reflectivity_resampled = Interp1D(well_times, well_reflectivity,
                                        method='linear', assume_sorted=True, extrap=0.0)(seismic_times)

        synthetic_trace = torch.nn.functional.conv1d(input=reflectivity_resampled.reshape(1, 1, -1),
                                                    weight=wavelet.reshape(1, 1, -1),
                                                    padding='same')
        return synthetic_trace.reshape(-1)


    def optimize_well_times_pytorch(self, n_segments=100, n_iters=1000,
                                    optimizer_params=None, regularization_params=None,
                                    limits=None, pbar='t', state=-1):
        """ !!. """
        #pylint: disable=import-outside-toplevel
        import torch
        from batchflow import Notifier

        limits = limits if limits is not None else slice(None)

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times, wavelet = previous_state['well_times'], previous_state['wavelet']

        # Convert data to PyTorch. Clone everything, as CPU tensors share data with numpy arrays
        seismic_times = torch.from_numpy(self.seismic_times).float().clone()
        seismic_trace = torch.from_numpy(self.seismic_trace).float().clone()
        well_reflectivity = torch.from_numpy(np.nan_to_num(self.well_reflectivity)).float().clone()

        well_times = torch.from_numpy(well_times).float().clone()
        wavelet = torch.from_numpy(wavelet).float().clone()
        dt = torch.from_numpy(np.diff(well_times, prepend=0)).float().clone()

        # Prepare variables for optimization: one multiplier for each segment
        # TODO: figure out a better way to multiplicate values
        if n_segments == len(well_times) or n_segments == 'well':
            multipliers = torch.ones(len(well_times), dtype=torch.float32, requires_grad=True)
            segment_size = 1
        else:
            multipliers = torch.ones(n_segments, dtype=torch.float32, requires_grad=True)
            segment_size = len(well_times) // n_segments + 1

        # Prepare infrastructure for train
        optimizer_params = {
            'lr': 0.0002,
            **(optimizer_params or {})
        }
        optimizer = torch.optim.AdamW((multipliers,), **optimizer_params)

        regularization_params = {
            'l1': 0.0, 'l2': 0.0,
            'dl1': 0.0, 'dl2': 0.0,
            **(regularization_params or {})
        }

        # Run train loop
        loss_history = []
        notifier = Notifier(pbar, frequency=min(50, n_iters),
                            monitors=[{'source': loss_history, 'format': 'correlation={:5.4f}'}])
        for _ in notifier(n_iters):
            # Loss # TODO: check that dt_tensor[0] is ~not updated!
            multipliers_ = torch.repeat_interleave(multipliers, segment_size)[:len(well_times)]
            multipliers_[0] = 1.0
            new_well_times = torch.cumsum(dt * multipliers_, dim=0)
            synthetic_trace = self.compute_resampled_synthetic_torch(well_times=new_well_times,
                                                                     well_reflectivity=well_reflectivity,
                                                                     seismic_times=seismic_times,
                                                                     wavelet=wavelet)
            loss = -self.correlation(seismic_trace, synthetic_trace)

            # Regularization
            dmultipliers = torch.diff(multipliers)
            regularization = (
                regularization_params['l1'] * torch.abs(multipliers - 1).mean() +
                regularization_params['l2'] * torch.abs((multipliers - 1) ** 2).mean() +
                regularization_params['dl1'] * torch.abs(dmultipliers).mean() +
                regularization_params['dl2'] * torch.abs(dmultipliers ** 2).mean()
            )

            # Update
            (loss + regularization).backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_history.append(-loss.detach().numpy().item())

        # Save state
        multipliers_ = torch.repeat_interleave(multipliers, segment_size)[:len(well_times)]
        multipliers_[0] = 1.0
        new_well_times = torch.cumsum(dt * multipliers_, dim=0).detach().numpy()
        correlation = self.compute_metric(well_times=new_well_times, wavelet=previous_state['wavelet'])
        state = {
            'type': 'optimize_well_times_pytorch',
            'well_times': new_well_times,
            'wavelet': previous_state['wavelet'],
            'correlation': correlation,
            'loss_history': loss_history,
            'multipliers': multipliers_.detach().numpy(),
        }
        self.states.append(state)


    # Wavelet optimization
    def optimize_wavelet(self, limits=None, state=-1, **kwargs):
        """ !!. """
        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-4, 'eps': 1e-7},
            **kwargs
        }

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times, wavelet = previous_state['well_times'], previous_state['wavelet']

        spectrum = np.fft.rfft(wavelet)
        # power_spectrum = np.abs(spectrum)
        # phase_spectrum = np.angle(spectrum)

        def minimization_proxy(phase_shift):
            new_wavelet = np.fft.irfft(spectrum * np.exp(1.0j * phase_shift), n=len(wavelet))
            return -self.compute_metric(well_times=well_times, wavelet=new_wavelet, limits=limits)

        optimization_results = minimize(minimization_proxy, x0=0., bounds=[[-np.pi/2, +np.pi/2]], **kwargs)

        phase_shift = optimization_results['x'].item()
        new_wavelet = np.fft.irfft(spectrum * np.exp(1.0j * phase_shift), n=len(wavelet))

        state = {
            'type': 'optimize_wavelet',
            'well_times': well_times,
            'wavelet': new_wavelet,
            'correlation': -optimization_results['fun'],
            'phase_shift': phase_shift,
            'phase_shift_angles': np.rad2deg(phase_shift),
        }
        self.states.append(state)

    def optimize_wavelet_(self, limits=None, n=10, delta=1., state=-1, **kwargs):
        """ !!. """
        kwargs = {
            'method': 'SLSQP',
            'options': {'maxiter': 100, 'ftol': 1e-4, 'eps': 1e-7},
            **kwargs
        }

        # Retrieve from previous state
        previous_state = state if isinstance(state, dict) else self.states[state]
        well_times = previous_state['well_times']
        wavelet = previous_state['wavelet']

        spectrum = np.fft.rfft(wavelet)
        power_spectrum = np.abs(spectrum)
        phase_spectrum = np.angle(spectrum)

        # Optimization objective
        def minimization_proxy(phase_shifts):
            new_phase_spectrum = phase_spectrum.copy()
            new_phase_spectrum[:len(phase_shifts)] = phase_shifts
            new_wavelet = np.fft.irfft(power_spectrum * np.exp(1.0j * new_phase_spectrum), n=len(wavelet))
            return -self.compute_metric(well_times=well_times, wavelet=new_wavelet, limits=limits)

        x0 = phase_spectrum[:n]
        optimization_bounds = np.array([x0-delta, x0+delta]).T
        optimization_bounds = np.clip(optimization_bounds, -np.pi, +np.pi)
        optimization_results = minimize(minimization_proxy, x0=x0, bounds=optimization_bounds, **kwargs)

        # Retrieve solution
        phase_shifts = optimization_results['x']
        new_phase_spectrum = phase_spectrum.copy()
        new_phase_spectrum[:len(phase_shifts)] = phase_shifts
        new_wavelet = np.fft.irfft(power_spectrum * np.exp(1.0j * new_phase_spectrum), n=len(wavelet))

        state = {
            'type': 'optimize_wavelet',
            'well_times': well_times,
            'wavelet': new_wavelet,
            'correlation': -optimization_results['fun'],
            'phase_shifts': phase_shifts,
        }
        self.states.append(state)

    # TODO:
    # SciPy full-vector optimization
    # PyTorch full-vector optimization
    # DTW full-vector optimization
    # Wavelet optimization: phase/multiphase
    # MSE optimization: find wavelet amplitude

    # Metrics
    def evaluate_markers(self, markers, state=-1):
        """ !!. """
        state = state if isinstance(state, dict) else self.states[state]

        if isinstance(markers, str):
            markers = pd.read_csv(markers, sep='\t')
        markers = markers[markers['Well'] == self.well.name]

        results_df = []
        for idx, row in markers.iterrows():
            marker_name = row['Top']
            if not marker_name.isupper():
                continue

            marker_depth = row['TVDSS [m]']
            marker_time = row['Time [s] X']
            idx = np.searchsorted(self.well.index, marker_depth)
            if self.well.index[idx] - marker_depth > marker_depth - self.well.index[idx-1]:
                idx -= 1
            predicted_time = state['well_times'][idx]

            results_df.append({
                'Top': marker_name,
                'TVDSS [m]': marker_depth,
                'Time [s]': marker_time,
                'Predicted Time [s]': round(predicted_time, 6),
                'Diff Time [s]': round(predicted_time - marker_time, 6),
            })

        return pd.DataFrame(results_df)


    # Export
    def save_well_times(self, path, state=-1):
        """ !!. """
        state = self.states[state]
        path = self.field.make_path(path, name=self.well.name)

        well_times = state['well_times']
        depths = self.well.index.values[self.well_bounds]

        data = np.array([depths, well_times]).T
        df = pd.DataFrame(data=data, columns=['MD, m', 'TWT, s'])

        df.to_csv(path, header=True, index=False, sep=' ')

    def save_wavelet(self, path, state=-1):
        """ !!. """
        state = self.states[state]
        path = self.field.make_path(path, name=self.well.name)

        wavelet = state['wavelet']
        times = np.arange(len(wavelet)) * self.field.sample_interval * 1e-3

        data = np.array([times, wavelet]).T
        df = pd.DataFrame(data=data, columns=['TWT, s', 'VALUE'])

        df.to_csv(path, header=True, index=False, sep=' ')

    def save_las(self, path, state=-1):
        """ !!. """
        state = self.states[-1]
        path = self.field.make_path(path, name=self.well.name)

        well_times = state['well_times']
        dt_optimized = np.diff(well_times, prepend=0) * 1e6
        well_impedance = self.well_impedance

        if well_times.size != self.well.shape[0]:
            pad_width = (self.well_bounds.start, self.well.shape[0]-self.well_bounds.stop)
            dt_optimized = np.pad(dt_optimized, pad_width=pad_width, constant_values=np.nan)
            well_impedance = np.pad(well_impedance, pad_width=pad_width, constant_values=np.nan)


        self.well.lasfile.append_curve('DT_OPTIMIZED', dt_optimized, unit='us/ft', descr='DT_OPTIMIZED')
        self.well.lasfile.append_curve('AI_USED', well_impedance, unit='kPa.s/m', descr='AI_USED')
        self.well.lasfile.write(path, version=2.0)

    def save_synthetic(self, path, state=-1):
        """ !!. """
        state = self.states[state]
        path = self.field.make_path(path, name=self.well.name)

        synthetic_trace = self.compute_resampled_synthetic(**state)
        synthetic_trace = synthetic_trace.reshape(1, 1, -1)

        array_to_segy(synthetic_trace, path=path, origin=(*self.coordinates, 0), pbar=False)


    # Visualization
    def show_state(self, state=-1, zoom=slice(None), force_dt=False, **kwargs):
        """ !!. """
        state = len(self.states) - 1 if state == -1 else state
        state_name = '<user_dict>' if isinstance(state, dict) else state
        state = state if isinstance(state, dict) else self.states[state]

        wavelet = state['wavelet']
        synthetic_trace = self.compute_resampled_synthetic(**state, multiply=True)
        correlation = self.compute_metric(synthetic_trace=synthetic_trace)

        dt = np.diff(self.well_times)
        dt_state = np.diff(state['well_times'])

        # Seismic to synthetic comparison; wavelet
        well_times = state['well_times'][1:]
        seismic_times = self.seismic_times[zoom]
        wavelet_times = self.seismic_times[:len(wavelet)]
        wavelet_times -= wavelet_times[len(wavelet)//2 + 0]

        data = [[(seismic_times, self.seismic_trace[zoom]), (seismic_times, synthetic_trace[zoom])],
                [(wavelet_times, wavelet)]]

        # Interval velocities: show only if changed
        if not np.allclose(dt, dt_state) or force_dt:
            iv = 0.3048 / dt
            iv_state = 0.3048 / dt_state
            iv_diff = np.abs(iv - iv_state)
            data.append([(well_times, iv), (well_times, iv_state), (well_times, iv_diff)])

            relative_iv = np.round(dt / dt_state, 2)
            data.append([(well_times, relative_iv)])

        kwargs = {
            'combine': 'overlay',
            'ncols': 2,
            'ratio': 0.3 if len(data) == 2 else 0.5,
            'suptitle': f'Well `{self.well.name}`\nstate={state_name}; {correlation=:3.3f}',
            'title': ['seismic vs synthetic', 'wavelet',
                      'interval velocity', 'relative increase in velocity: dt/dt_state'],
            'xlabel': ['seismic time, s', 'time, s',
                       'well time, s', 'well time, s'],
            'ylabel': ['amplitude', 'amplitude', 
                       'velocity, m/s', 'ratio'],
            'label': [['seismic_trace', 'synthetic_trace'], '', 
                      ['original IV', 'state IV', 'diff IV'], ''],
            'xlabel_size': 18,
            **kwargs
        }
        plotter = plot(data, mode='curve', **kwargs)
        if len(data) == 4:
            plotter.subplots[-1].ax.axhline(1, linestyle='dashed', alpha=.5, color='sandybrown', linewidth=3)

        return plotter


    def show_wavelet(self, state=-1, **kwargs):
        """ !!. """
        state = len(self.states) - 1 if state == -1 else state
        state_name = '<user_dict>' if isinstance(state, dict) else state
        state = state if isinstance(state, dict) else self.states[state]
        wavelet = state['wavelet']
        correlation = state['correlation']

        spectrum = np.fft.rfft(wavelet)
        power_spectrum = np.abs(spectrum)
        phase_spectrum = np.angle(spectrum)
        frequencies = np.fft.rfftfreq(len(wavelet), d=self.field.sample_interval * 1e-3)

        kwargs = {
            'combine': 'separate',
            'ncols': 3,
            'ratio': 0.25,
            'suptitle': f'well `{self.well.name}`\nstate={state_name}; {correlation=:3.3f}',
            'title': ['wavelet', 'power spectrum', 'phase spectrum'],
            **kwargs
        }
        return plot([wavelet, (frequencies, power_spectrum), (frequencies, phase_spectrum)], mode='curve', **kwargs)


    def show_progress(self, start_idx=1, **kwargs):
        """ !!. """
        data = [[state.get('correlation', self.compute_metric(**state)) for state in self.states[start_idx:]]]

        kwargs = {
            'combine': 'separate',
            'title': ['correlation over states', 'time shifts of states'],
            'xlabel': 'state index',
            'ylabel': ['correlation', 'time shift (ms)'],
            **kwargs
        }

        # Extrema optimization states
        states = [state for state in self.states if state['type'] == 'optimize_extrema']
        if states:
            time_shifts = [state['time_shift'] * 1000 for state in states]
            data.append(time_shifts)

        plotter = plot(data, mode='curve', ncols=2 if states else 1, **kwargs)

        if states:
            plotter[1].ax.lines.pop(0)
            colors = np.where(np.array(time_shifts) > 0, 'r', 'b')
            plotter[1].ax.bar(range(len(states)), time_shifts, color=colors)
        return plotter


    def show_crosscorrelation(self, state=1, n_peaks=3, **kwargs):
        """ !!. """
        state = state if isinstance(state, dict) else self.states[state]
        if state['type'] not in {'compute_t0'}:
            raise TypeError('!!.')

        kwargs = {
            'title': 'correlation VS shift of well data',
            'xlabel': 'shift, seconds', 'ylabel': 'correlation',
            'fontsize': 18, 'title_size': 22,
            **kwargs
        }

        plotter = plot((state['shifts'], state['values']), mode='curve', **kwargs)
        plotter[0].ax.scatter(state['peak_shifts'], state['peak_values'], s=15, c='r', marker='8')

        for idx in range(n_peaks):
            shift = state['peak_shifts'][idx]
            correlation = state['peak_values'][idx]
            plotter[0].ax.axvline(shift, correlation, linestyle='dashed', color='orange', alpha=0.9,
                                  label=f'{shift=:+2.3f} {correlation=:2.3f}')
        plotter[0].ax.legend(prop={'size': 14})
        return plotter

    def show_time_shifts(self, zoom=None, **kwargs):
        """ !!. """
        zoom = zoom if zoom is not None else (0, self.seismic_times[-1])
        kwargs = {
            'title': 'Extrema shift visualization',
            'xlim': zoom,
            'ylabel': '', 'ylim': (0, 1), 'ytick_labels': '',
            **kwargs
        }

        plotter = plot((self.seismic_times, self.seismic_times * np.nan), mode='curve', **kwargs)

        states = [state for state in self.states if state['type'] == 'optimize_extrema']
        for i, state in enumerate(states):
            time_before, time_after = state['time_before'], state['time_after']
            plotter[0].ax.axvline(time_before, linestyle='--', alpha=0.8, color='blue')
            plotter[0].ax.axvline(time_after, linestyle='solid', alpha=1, color='green')

            text = f'{state["time_shift"] * 1000:+2.3f} ms'
            plotter[0].ax.annotate(text, xy=(max(time_before, time_after), (i + 1) / len(states)), size=14)
        return plotter
