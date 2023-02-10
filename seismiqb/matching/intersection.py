""" Intersections between fields. """
from textwrap import dedent

import numpy as np
import scipy
import matplotlib.pyplot as plt

from .functional import compute_correlation, compute_r2, modify_trace, minimize_proxy
from ..plotters import plot



class Intersection:
    """ Base class to describe an intersection between two fields. """
    @classmethod
    def new(cls, field_0, field_1, limits=slice(None), pad_width=20, threshold=10, unwrap=True):
        """ Create one or more instances of intersection classes with automatic class selection.
        Preferred over directly instantiating objects.
        """
        is_2d_0 = 1 in field_0.spatial_shape
        is_2d_1 = 1 in field_1.spatial_shape
        if is_2d_0 and is_2d_1:
            return Intersection2d2d.new(field_0, field_1, limits=limits, pad_width=pad_width,
                                        threshold=threshold, unwrap=unwrap)

        return Intersection2d3d(field_0, field_1)


class Intersection2d2d:
    """ Intersection between two 2D fields. """
    @classmethod
    def new(cls, field_0, field_1, limits=slice(None), pad_width=20, threshold=10, unwrap=True):
        """ Create one or more instances of intersection.
        Preferred over directly instantiating objects.
        """
        values_0 = field_0.geometry.headers[['CDP_X', 'CDP_Y']].values
        values_1 = field_1.geometry.headers[['CDP_X', 'CDP_Y']].values

        bbox_0 = np.sort(values_0[[0, -1]].T, axis=-1)
        bbox_1 = np.sort(values_1[[0, -1]].T, axis=-1)

        overlap = np.maximum(bbox_0[:, 0], bbox_1[:, 0]), np.minimum(bbox_0[:, 1], bbox_1[:, 1])
        if (overlap[1] - overlap[0]).min() < 0:
            return False if unwrap else []

        # pylint: disable=import-outside-toplevel
        from shapely import LineString, MultiLineString, MultiPoint, GeometryCollection
        # TODO: improve and describe edge cases
        line_0 = LineString(values_0)
        line_1 = LineString(values_1)

        intersection = line_0.intersection(line_1)
        if isinstance(intersection, (MultiLineString, MultiPoint, GeometryCollection)):
            # intersection = intersection.geoms[0]
            points = [list(zip(*geometry.xy)) for geometry in intersection.geoms]
            points = sum(points, [])
        else:
            points = list(zip(*intersection.xy))

        result = []
        for point in points:
            trace_idx_0 = ((values_0 - point) ** 2).sum(axis=-1).argmin()
            trace_idx_1 = ((values_1 - point) ** 2).sum(axis=-1).argmin()

            instance = cls(field_0=field_0, field_1=field_1,
                           trace_idx_0=trace_idx_0, trace_idx_1=trace_idx_1,
                           limits=limits, pad_width=pad_width)

            if not any(other.is_similar(instance, threshold=threshold) for other in result):
                result.append(instance)

        if not result:
            return False if unwrap else []
        return result[0] if len(result) == 1 and unwrap else result


    def __init__(self, field_0, field_1, trace_idx_0, trace_idx_1, limits=slice(None), pad_width=20):
        self.field_0, self.field_1 = field_0, field_1
        self.trace_idx_0, self.trace_idx_1 = trace_idx_0, trace_idx_1
        self.limits = limits
        self.pad_width = pad_width
        self.max_depth = max(field_0.depth, field_1.depth)

        # Compute distance
        self.coordinates_0 = field_0.geometry.headers[['CDP_X', 'CDP_Y']].values[trace_idx_0]
        self.coordinates_1 = field_1.geometry.headers[['CDP_X', 'CDP_Y']].values[trace_idx_1]
        self.distance = ((self.coordinates_0 - self.coordinates_1).astype(np.float64) ** 2).sum() ** (1 / 2)

        self.matching_results = None

    def is_similar(self, other, threshold=10):
        """ Check if other intersection is close index-wise. """
        if abs(self.trace_idx_0 - other.trace_idx_0) <= threshold and \
            abs(self.trace_idx_1 - other.trace_idx_1) <= threshold:
            return True
        return False


    def to_dict(self, precision=3):
        """ Represent intersection parameters (including computed matching values) as a dictionary. """
        intersection_dict = {
            'field_0': self.field_0.name,
            'field_1': self.field_1.name,
            'distance': self.distance,
        }

        if self.matching_results is not None:
            intersection_dict.update(self.matching_results)

        for key, value in intersection_dict.items():
            if isinstance(value, (float, np.floating)):
                intersection_dict[key] = round(value, precision)
        return intersection_dict


    # Data
    def prepare_traces(self, limits=None, index_shifts=(0, 0), pad_width=0, n=1):
        """ Prepare traces from both intersecting fields.
        Under the hood, we load traces, pad to max depth, slice with `limits` and add additional `pad_width`.
        Also, we average over `n` traces at loading to reduce noise.
        """
        limits = limits or self.limits
        pad_width = pad_width or self.pad_width

        trace_0 = self._prepare_trace(self.field_0, index=self.trace_idx_0 + index_shifts[0],
                                      limits=limits, pad_width=pad_width, n=n)
        trace_1 = self._prepare_trace(self.field_1, index=self.trace_idx_1 + index_shifts[1],
                                      limits=limits, pad_width=pad_width, n=n)
        return trace_0, trace_1

    def _prepare_trace(self, field, index, limits=None, pad_width=0, n=1):
        # Load data
        nhalf = (n - 1) // 2
        indices = list(range(index - nhalf, index + nhalf + 1))
        traces = field.geometry.load_by_indices(indices)
        trace = np.mean(traces, axis=0)

        # Pad/slice
        if trace.size < self.max_depth:
            trace = np.pad(trace, (0, self.max_depth - trace.size))
        trace = trace[limits]
        if pad_width > 0:
            trace = np.pad(trace, pad_width)
        return trace


    # Matching algorithms
    def match_traces(self, method='analytic', **kwargs):
        """ Selector for matching method.
        Refer to the documentation of :meth:`match_traces_analytic` and :meth:`match_traces_optimize` for details.

        TODO: add `mixed` mode, where we select the initial point by `analytic` method and then use optimization
        procedure to find the exact location.
        """
        if method in {'analytic'}:
            matching_results = self.match_traces_analytic(**kwargs)
        else:
            matching_results = self.match_traces_optimize(**kwargs)

        matching_results['petrel_corr'] = (matching_results['corr'] + 1) / 2
        self.matching_results = matching_results
        return matching_results


    def match_traces_optimize(self, limits=None, index_shifts=(0, 0), pad_width=0, n=1,
                              init_shifts=range(-100, +100), init_angles=(0,), metric='correlation',
                              bounds_shift=(-150, +150), bounds_angle=None, bounds_gain=(0.9, 1.1),
                              maxiter=100, eps=1e-6):
        """ Match traces by iterative optimization of the selected loss function.
        Slower, than :meth:`match_traces_analytic`, but allows for finer control.

        We use every combination of parameters in `init_shifts` and `init_angles` as
        the starting point for optimization. This way, we try to avoid local minima, improving the result by a lot.
        The optimization is bounded: `bounds_*` parameters allow to control the spread of possible values.
        """
        # Load data
        trace_0, trace_1 = self.prepare_traces(limits=limits, index_shifts=index_shifts, pad_width=pad_width, n=n)

        # For each element in init, perform optimize
        minimize_results = []
        for init_shift in init_shifts:
            for init_angle in init_angles:
                bounds_angle_ = bounds_angle or (init_angle-eps, init_angle+eps)
                minimize_result = scipy.optimize.minimize(fun=minimize_proxy,
                                                          x0=np.array([init_shift, init_angle, 1.0]),
                                                          args=(trace_0, trace_1, metric),
                                                          bounds=(bounds_shift, bounds_angle_, bounds_gain),
                                                          method='SLSQP',
                                                          options={'maxiter': maxiter,
                                                                   'ftol': 1e-6, 'eps': 1e-3})
                minimize_results.append(minimize_result)
        minimize_results = np.array([(item.fun, *item.x) for item in minimize_results])

        # Find the best result
        argmin = np.argmin(minimize_results[:, 0])
        best_loss, best_shift, best_angle, best_gain = minimize_results[argmin]
        best_corr = compute_correlation(trace_0,
                                        modify_trace(trace_1, shift=best_shift, angle=best_angle, gain=best_gain))

        return {
            'corr': best_corr,
            'shift': best_shift,
            'angle': best_angle,
            'gain': best_gain,
            'loss': best_loss,
        }


    def match_traces_analytic(self, limits=None, index_shifts=(0, 0), pad_width=0, n=1,
                              twostep=False, twostep_margin=10,
                              max_shift=100, resample_factor=10, metric='correlation',
                              apply_correction=False, correction_step=3, return_intermediate=False):
        """ Match traces by using analytic formulae.
        Bishop, Nunns "`Correcting amplitude, time, and phase mis-ties in seismic data
        <https://www.researchgate.net/publication/249865260>`_"
        Fast, but rather unflexible.

        Under the hood, the algorithm works as follows:
            - we compute possible shifts with possibly non-whole numbers (`resample_factor`)
            - compute correlation for each possible shift
            - compute envelope and instantaneous phase of the cross-correlation
            - argmax of the envelope is the optimal shift, and the phase at this shift is the optimal angle.
            Essentially, this is equivalent to finding the best combination of the trace and its analytic counterpart.

        # TODO: add optional `fft` to speed up computation; add better `gain` computation
        """
        # Load data
        trace_0, trace_1 = self.prepare_traces(limits=limits, index_shifts=index_shifts, pad_width=pad_width, n=n)

        # Prepare array of tested shifts
        if twostep:
            # Compute approximate `shift` to narrow the interval
            shifts = np.linspace(-max_shift, max_shift, 2*max_shift + 1, dtype=np.float32)
            shift = self._match_traces_analytic(trace_0=trace_0, trace_1=trace_1, shifts=shifts,
                                                metric=metric, apply_correction=apply_correction,
                                                correction_step=correction_step,
                                                return_intermediate=False)['shift']
            shifts = np.linspace(shift - twostep_margin, shift + twostep_margin,
                                 2*twostep_margin*resample_factor + 1, dtype=np.float32)
        else:
            shifts = np.linspace(-max_shift, max_shift, 2*max_shift*resample_factor + 1, dtype=np.float32)

        # Compute `shift` with required precision
        matching_results = self._match_traces_analytic(trace_0=trace_0, trace_1=trace_1, shifts=shifts,
                                                       metric=metric, apply_correction=apply_correction,
                                                       correction_step=correction_step,
                                                       return_intermediate=return_intermediate)

        matching_results['corr'] = self.evaluate(shift=matching_results['shift'],
                                                 angle=matching_results['angle'],
                                                 gain=matching_results['gain'],
                                                 pad_width=pad_width, limits=limits, n=n)
        return matching_results

    def _match_traces_analytic(self, trace_0, trace_1, shifts, metric='correlation',
                               apply_correction=False, correction_step=3, return_intermediate=False):
        # Compute metrics for each shift on a resampled grid
        # TODO: can be significantly sped up by hard-coding correlation here
        metric_function = compute_correlation if metric == 'correlation' else compute_r2
        metrics = [metric_function(trace_0, modify_trace(trace_1, shift=shift)) for shift in shifts]
        metrics = np.array(metrics)

        # Compute envelope and phase of metrics
        analytic_signal = scipy.signal.hilbert(metrics)
        envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_phase = np.rad2deg(instantaneous_phase)

        # Find the best shift and compute its relative quality
        idx = np.argmax(envelope)

        # Optional correction: parabolic interpolation in the neighborhood of a maxima
        if apply_correction is False:
            best_shift = shifts[idx]
            best_angle = instantaneous_phase[idx]
            best_gain = 1 # np.linalg.norm(trace_1) / np.linalg.norm(trace_0) #TODO
        else:
            correction = ((metrics[idx-correction_step] - metrics[idx+correction_step]) /
                          (2*metrics[idx-correction_step] - 4*metrics[idx] + 2*metrics[idx+correction_step]))
            quality = metrics[idx]\
                      - 0.25 * (((metrics[idx-correction_step] - metrics[idx+correction_step]) * correction) /
                                (np.linalg.norm(trace_0) * np.linalg.norm(trace_1)))
            _ = quality

            # Shift: correct according to values to the sides of maximum
            corrected_idx = int(idx + correction)
            best_shift = shifts[corrected_idx]

            # Angle: correct according to values to the sides of maximum
            p0 = instantaneous_phase[idx]
            p1 = instantaneous_phase[idx+correction_step] if correction >= 0 else \
                 instantaneous_phase[idx-correction_step]
            if p1 - p0 > 180:
                p1 = p1 - 360
            elif p1 - p0 < -180:
                p1 = p1 + 360
            best_angle = p0 + ((p1 - p0) * correction if correction >= 0 else (p0 - p1) * correction)

            # Gain: no correction
            best_gain = np.linalg.norm(trace_1) / np.linalg.norm(trace_0)

        matching_results = {
            'shift': best_shift,
            'angle': best_angle,
            'gain': best_gain,
        }

        if return_intermediate:
            matching_results.update({
                'trace_0': trace_0,
                'trace_1': trace_1,
                'shifts': shifts,
                'metrics': metrics,
                'envelope': envelope,
                'instantaneous_phase': instantaneous_phase,
            })
        return matching_results


    def evaluate(self, shift=0, angle=0, gain=1, metric='correlation', pad_width=0, limits=None, n=1):
        """ Compute provided metric with a given mistie parameters. """
        trace_0, trace_1 = self.prepare_traces(pad_width=pad_width, limits=limits, n=n)
        metric_function = compute_correlation if metric == 'correlation' else compute_r2
        return metric_function(trace_0, modify_trace(trace_1, shift=shift, angle=angle, gain=gain))


    # Visualization
    def __str__(self):
        return dedent(f"""
        Intersection of "{self.field_0.short_name}.sgy" and "{self.field_1.short_name}.sgy"
        distance                     {self.distance:4.2f} m
        trace_idx_0                  {self.trace_idx_0}
        trace_idx_1                  {self.trace_idx_1}
        coordinates_0                {self.coordinates_0.tolist()}
        coordinates_1                {self.coordinates_1.tolist()}
        """).strip()

    def show_curves(self, method='analytic', limits=None, index_shifts=(0, 0), pad_width=0, n=1,
                    max_shift=100, resample_factor=10, metric='correlation', apply_correction=False, **kwargs):
        """ Display traces, cross-correlation vs shift and phase vs shift graphs. """
        # Get matching results with all the intermediate variables
        matching_results = self.match_traces(method=method,
                                             limits=limits, index_shifts=index_shifts, pad_width=pad_width, n=n,
                                             max_shift=max_shift, resample_factor=resample_factor,
                                             metric=metric, apply_correction=apply_correction,
                                             return_intermediate=True)

        trace_0, trace_1 = matching_results['trace_0'], matching_results['trace_1']
        shifts, metrics, envelope, instantaneous_phase = (matching_results['shifts'],
                                                          matching_results['metrics'],
                                                          matching_results['envelope'],
                                                          matching_results['instantaneous_phase'])

        # Prepare plotter parameters
        limits = limits or self.limits
        pad_width = pad_width or self.pad_width
        start_tick = (limits.start or 0) - pad_width
        ticks = np.arange(start_tick, start_tick + len(trace_0))

        kwargs = {
            'title': ['traces', 'cross-correlation', 'instantaneous phase'],
            'label': [['trace_0', 'trace_1'], ['crosscorrelation', 'envelope'], 'instant phases'],
            'xlabel': ['depth', 'shift', 'shift'], 'xlabel_size': 16,
            'ylabel': ['amplitude', 'metric', 'phase (degrees)'],
            'xlim': [(start_tick, start_tick + len(trace_0)),
                     (-max_shift, +max_shift),
                     (-max_shift, +max_shift)],
            'ratio': 0.8,
            **kwargs
        }

        plotter = plot([[(ticks, trace_0), (ticks, trace_1)],
                        [(shifts, metrics), (shifts, envelope)],
                        [(shifts, instantaneous_phase)]],
                       mode='curve', **kwargs)

        # Add more annotations
        shift = matching_results['shift']
        angle = matching_results['angle']
        corr = matching_results['corr']
        plotter[1].ax.axvline(shift, linestyle='--', alpha=0.9, color='green')
        plotter[1].add_legend(mode='curve', label=f'optimal shift: {shift:4.3f}',
                              alpha=0.9, color='green')
        plotter[1].ax.axhline(corr, linestyle='--', alpha=0.9, color='red')
        plotter[1].add_legend(mode='curve', label=f'max correlation: {corr:4.3f}',
                              alpha=0.9, color='red')
        plotter[2].ax.axvline(shift, linestyle='--', alpha=0.9, color='green')
        plotter[2].add_legend(mode='curve', label=f'optimal angle: {angle:4.3f}')
        return plotter


    def show_lines(self, figsize=(14, 8), colors=('b', 'r'), arrow_step=20, arrow_size=30):
        """ Display shot lines on a 2d graph in CDP coordinates. """
        fig, ax = plt.subplots(figsize=figsize)

        # Data
        for field, color in zip([self.field_0, self.field_1], colors):
            values = field.geometry.headers[['CDP_X', 'CDP_Y']].values
            x, y = values[:, 0], values[:, 1]
            ax.plot(x, y, color, label=field.short_name)

            idx = x.size // 2
            ax.annotate('', size=arrow_size,
                        xytext=(x[idx-arrow_step], y[idx-arrow_step]),
                            xy=(x[idx+arrow_step], y[idx+arrow_step]),
                        arrowprops=dict(arrowstyle="->", color=color))

        # Annotations
        ax.set_title(f'"{self.field_0.short_name}.sgy" and "{self.field_1.short_name}.sgy"', fontsize=26)
        ax.set_xlabel('CDP_X', fontsize=22)
        ax.set_ylabel('CDP_Y', fontsize=22)
        ax.legend(prop={'size' : 22})
        ax.grid()
        fig.show()


    def show_metric_surface(self, metric='correlation', limits=None, index_shifts=(0, 0), pad_width=0, n=1,
                            shifts=range(-20, +20+1, 1), angles=range(-180, +180+1, 30),
                            figsize=(14, 8), cmap='seismic', levels=7, grid=True):
        """ Display metric values as a function of shift and angle. """
        # Compute metric matrix: metric value for each combination of shift and angle
        trace_0, trace_1 = self.prepare_traces(limits=limits, index_shifts=index_shifts, pad_width=pad_width, n=n)
        metric_function = compute_correlation if metric == 'correlation' else compute_r2

        metric_matrix = np.empty((len(shifts), len(angles)))
        for i, shift in enumerate(shifts):
            for j, angle in enumerate(angles):
                modified_trace_1 = modify_trace(trace_1, shift=shift, angle=angle)

                metric_matrix[i, j] = metric_function(trace_0, modified_trace_1)

        # Show contourf and contour
        fig, ax = plt.subplots(1, figsize=figsize)
        img = ax.contourf(angles, shifts, metric_matrix, cmap=cmap, levels=levels)
        fig.colorbar(img)

        contours = ax.contour(angles, shifts, metric_matrix, levels=levels, colors='k', linewidths=0.4)
        ax.clabel(contours, contours.levels, inline=True, fmt=lambda x: f'{x:2.1f}', fontsize=10)

        ax.set_title('METRIC SURFACE', fontsize=20)
        ax.set_xlabel('PHASE (DEGREES)', fontsize=16)
        ax.set_ylabel('SHIFT (MS)', fontsize=16)
        ax.grid(grid)
        fig.show()


    def show_neighborhood(self, max_index_shift=7, limits=None, pad_width=0, n=1, max_shift=10, resample_factor=10):
        """ Compute matching on all neighboring traces. """
        # Prepare data
        k = max_index_shift * 2 + 1
        matrix = np.empty((k, k), dtype=np.float32)
        iterator = range(-max_index_shift, max_index_shift + 1)
        for i, index_shift_1 in enumerate(iterator):
            for j, index_shift_2 in enumerate(iterator):
                matching_results = self.match_traces_analytic(index_shifts=(index_shift_1, index_shift_2),
                                                              limits=limits, pad_width=pad_width, n=n,
                                                              max_shift=max_shift, resample_factor=resample_factor)
                matrix[i, j] = matching_results['corr']

        # Visualize
        value = matrix[max_index_shift, max_index_shift]
        delta = max(matrix.max() - value, value - matrix.min())
        vmin, vmax = value - delta, value + delta

        return plot(matrix, colorbar=True, cmap='seismic',
                    title='Correlation values for neighbouring indices of intersection',
                    vmin=vmin, vmax=vmax,
                    extent=(-max_index_shift, +max_index_shift,
                            -max_index_shift, +max_index_shift))


    def show_composite_slide(self, sides=(0, 0), limits=None, gap_width=1, pad_width=0,
                             shift=0, angle=0, gain=1, **kwargs):
        """ Display sides of shot lines on one plot. """
        side_0, side_1 = sides
        limits = limits or self.limits
        pad_width = pad_width or self.pad_width

        # Load data
        slide_0 = self.field_0.load_slide(0)
        slide_0 = slide_0[:self.trace_idx_0 + 1] if side_0 == 0 else slide_0[self.trace_idx_0:]

        slide_1 = self.field_1.load_slide(0)
        slide_1 = slide_1[:self.trace_idx_1 + 1] if side_1 == 0 else slide_1[self.trace_idx_1:]

        # Pad to adjust for field delays
        slide_0 = (np.pad(slide_0, ((0, 0), (self.field_0.delay, 0)))
                   if self.field_0.delay > 0 else slide_0[:, -self.field_0.delay:])
        slide_1 = (np.pad(slide_1, ((0, 0), (self.field_1.delay, 0)))
                   if self.field_1.delay > 0 else slide_1[:, -self.field_1.delay:])

        # Pad to the same depth
        slide_0 = np.pad(slide_0, ((0, 0), (0, self.max_depth - slide_0.shape[1])))
        slide_1 = np.pad(slide_1, ((0, 0), (0, self.max_depth - slide_1.shape[1])))

        # Slice to limits
        slide_0 = slide_0[:, limits]
        slide_1 = slide_1[:, limits]

        # Additional padding
        slide_0 = np.pad(slide_0, ((0, 0), (pad_width, pad_width)))
        slide_1 = np.pad(slide_1, ((0, 0), (pad_width, pad_width)))

        # Apply modifications to the right side
        for c in range(slide_1.shape[0]):
            slide_1[c] = modify_trace(slide_1[c], shift=shift, angle=angle, gain=gain)

        combined_slide = np.concatenate([slide_0,
                                         np.zeros((gap_width, slide_0.shape[1])),
                                         slide_1], axis=0)

        # Compute correlation on traces
        trace_0 = slide_0[-1] if side_0 == 0 else slide_0[0]
        trace_1 = slide_1[-1] if side_1 == 0 else slide_1[0]
        correlation = compute_correlation(trace_0, trace_1)

        # Prepare plotter parameters
        start_tick = (limits.start or 0) - pad_width
        extent = (0, combined_slide.shape[0], start_tick + combined_slide.shape[1], start_tick)

        title = (f'"{self.field_0.short_name}.sgy" x "{self.field_1.short_name}.sgy"\n'
                 f'{shift=:3.2f} {angle=:3.1f} {gain=:3.2f}\n'
                 f'{correlation=:3.2f}\n'
                 f'corrected_correlation={(1 + correlation)/2:3.2f}')
        kwargs = {
            'cmap': 'Greys_r',
            'colorbar': True,
            'title': title,
            'extent': extent,
            **kwargs
        }
        return plot(combined_slide, **kwargs)


class Intersection2d3d:
    """ TODO. """
