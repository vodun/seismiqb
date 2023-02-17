""" Collection of multiple intersecting fields. """
import numpy as np
import pandas as pd
from numba import njit

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from batchflow.notifier import Notifier
from .intersection import Intersection
from ..plotters import plot



class FieldCollection:
    """ Collection of 2D fields and their intersections. """
    def __init__(self, fields, limits=None, pad_width=0, threshold=10, n_intersections=np.inf):
        self.fields = fields
        self.n_fields = len(fields)

        self.intersections = {}
        for i, field_0 in enumerate(fields[:-1]):
            for j, field_1 in enumerate(fields[i+1:], start=i+1):
                intersections = Intersection.new(field_0=field_0, field_1=field_1,
                                                 limits=limits, pad_width=pad_width, threshold=threshold, unwrap=False)

                for k, intersection in enumerate(intersections):
                    if k > n_intersections:
                        break
                    key = (i, j, k)
                    self.intersections[key] = intersection
                    intersection.key = (i, j, k)

        self.corrections = {}


    # Work with intersections
    def find_intersection(self, name_0, name_1):
        """ Find intersection by names of shot lines. """
        for intersection in self.intersections.values():
            if (name_0 in intersection.field_0.name and name_1 in intersection.field_1.name) or \
                (name_1 in intersection.field_0.name and name_0 in intersection.field_1.name):
                return intersection
        raise KeyError(f'No intersection of `{name_0}` and `{name_1}` in collection!')

    def match_intersections(self, pbar='t', method='analytic', limits=None, pad_width=None, n=1, transform=None,
                            **kwargs):
        """ Match traces on each intersection. """
        for intersection in Notifier(pbar)(self.intersections.values()):
            intersection.match_traces(method=method, limits=limits, pad_width=pad_width, n=n, transform=transform,
                                      **kwargs)

    def get_matched_value(self, key):
        """ Get required `key` value from each of the intersections. """
        return [intersection.matching_results[key] for intersection in self.intersections.values()]

    def intersections_df(self, errors=False, corrections=False, indices=False):
        """ Dataframe with intersections: each row describes quality of matching, mis-tie parameters for every crossing.
        If corrections are available, also use them.
        """
        df = []
        for key, intersection in self.intersections.items():
            intersection_dict = {'key': key, **intersection.to_dict()}
            df.append(intersection_dict)
        df = pd.DataFrame(df)
        df.set_index('key', inplace=True)

        # Corrections are distributed
        if self.corrections:
            shifts_errors = np.abs(self.corrections['shift']['errors'])
            gains_errors = np.abs(self.corrections['gain']['errors'])
            angles_errors = np.abs(self.corrections['angle']['errors'])
            suspicious = ((shifts_errors > shifts_errors.mean() + 3 * shifts_errors.std()) +
                          (gains_errors > gains_errors.mean() + 3 * gains_errors.std()) +
                          (angles_errors > angles_errors.mean() + 3 * angles_errors.std()))

            if errors:
                df['shifts_errors'] = shifts_errors
                df['gains_errors'] = gains_errors
                df['angles_errors'] = angles_errors
            df['suspicious'] = suspicious

            if corrections:
                idx_0 = [key[0] for key in self.intersections]
                df['shift_correction'] = self.corrections['shift']['x'][idx_0]
                df['angle_correction'] = self.corrections['angle']['x'][idx_0]
                df['gain_correction'] = self.corrections['gain']['x'][idx_0]

        columns = [
            'field_0_name', 'field_1_name', 'distance', 'suspicious',
            'corr', 'petrel_corr',
            'shift', 'shift_correction', 'angle', 'angle_correction', 'gain', 'gain_correction',
        ]
        columns = [c for c in columns if c in df.columns.values]
        columns += list(set(df.columns.values) - set(columns))
        df = df[columns]
        return df


    # Work with fields
    def distribute_corrections(self, skip_index=-1, max_iters=100, alpha=0.75, tolerance=0.00001):
        """ Distribute computed mis-ties from each intersection to fields.
        Under the hood, we iteratively optimize mis-ties of every type with respect to ~MSE loss.

        For the phase corrections, we also add phase unwrapping. Refer to the original article for details.
        Bishop, Nunns "`Correcting amplitude, time, and phase mis-ties in seismic data
        <https://www.researchgate.net/publication/249865260>`_"
        """
        a = np.array([key[:2] for key in self.intersections])
        n = self.n_fields

        # Shift
        b = np.array(self.get_matched_value('shift'))
        x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                     max_iters=max_iters, alpha=alpha, tolerance=tolerance)
        xk, xl = x[a[:, 0]], x[a[:, 1]]
        errors = b - (xk - xl)
        self.corrections['shift'] = {'x': x, 'errors': errors, 'loss': loss}

        # Gain
        b = np.array(self.get_matched_value('gain'))
        x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                     max_iters=max_iters, alpha=alpha, tolerance=tolerance)
        xk, xl = x[a[:, 0]], x[a[:, 1]]
        errors = b - (xk - xl)
        self.corrections['gain'] = {'x': x, 'errors': errors, 'loss': loss}

        # Angle
        b = np.array(self.get_matched_value('angle'))
        x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                     max_iters=max_iters, alpha=alpha, tolerance=tolerance)

        b_arange = np.arange(len(b))

        for _ in range(0, len(self.intersections)):
            xk, xl = x[a[:, 0]], x[a[:, 1]]

            b_unwrapped = np.repeat(b[:, np.newaxis], 3, axis=-1) + [-360, 0, +360]
            errors_unwrapped = np.abs(b_unwrapped - (xk - xl).reshape(-1, 1))
            argmins = np.argmin(errors_unwrapped, axis=-1)

            # Stop condition: no phase unwrapping required
            if (argmins == 1).all():
                # TODO: add one-time forced perturbation
                break

            b = b_unwrapped[b_arange, argmins]
            x, loss = distribute_misties(a=a, b=b, n=n, skip_index=skip_index,
                                         max_iters=max_iters, alpha=alpha, tolerance=tolerance)

        errors = b - (xk - xl)
        self.corrections['angle'] = {'x': x, 'errors': errors, 'loss': loss, 'b': b}
        # TODO: add return with info about the process


    def compute_suspicious(self):
        """ For each intersection, compute whether it is suspicious.
        # TODO: add more checks
        """
        if self.corrections is None:
            return [False] * len(self.intersections)
        shifts_errors = np.abs(self.corrections['shift']['errors'])
        gains_errors = np.abs(self.corrections['gain']['errors'])
        angles_errors = np.abs(self.corrections['angle']['errors'])
        suspicious = ((shifts_errors > shifts_errors.mean() + 3 * shifts_errors.std()) +
                      (gains_errors > gains_errors.mean() + 3 * gains_errors.std()) +
                      (angles_errors > angles_errors.mean() + 3 * angles_errors.std()))
        return suspicious

    def remove_suspicious(self, skip_index=-1, max_iters=100, alpha=0.75, tolerance=0.00001):
        """ Remove all suspicious intersections and re-distribute corrections. """
        suspicious = self.compute_suspicious()
        indices = np.nonzero(suspicious)[0]
        keys = list(self.intersections.keys())
        for idx in indices[::-1]:
            key = keys[idx]
            self.intersections.pop(key)

        self.distribute_corrections(skip_index=skip_index, max_iters=max_iters, alpha=alpha, tolerance=tolerance)
        return indices


    def fields_df(self):
        """ Dataframe with fields: each row describes a field with computed mis-ties. """
        shifts = self.corrections['shift']['x']
        gains = self.corrections['gain']['x']
        angles = self.corrections['angle']['x']

        df = []
        for i, field in enumerate(self.fields):
            intersections = []
            recomputed_corrs = []

            for key, intersection in self.intersections.items():
                if i in key[:2]:
                    recomputed_corr = intersection.evaluate(shift=shifts[key[0]] - shifts[key[1]],
                                                            angle=angles[key[0]] - angles[key[1]],)
                    recomputed_corr = (recomputed_corr + 1) / 2
                    recomputed_corrs.append(recomputed_corr)

                    intersections.append(intersection)

            # Stats on intersections: no distribution of corrections
            dicts = [intersection.matching_results for intersection in intersections]
            corrs = [d['petrel_corr'] for d in dicts]

            # Stats on distributed corrections
            # TODO

            correction_results = {
                'name': field.name,
                'shift': shifts[i],
                'gain': gains[i],
                'angle': angles[i],

                'mean_recomputed_corr': np.mean(recomputed_corrs).round(3),
                'std_recomputed_corr': np.std(recomputed_corrs).round(3),

                'n_intersections': len(intersections),
                'mean_corr_intersections': np.mean(corrs).round(3),
                'std_corr_intersections': np.std(corrs).round(3),
            }
            field.correction_results = correction_results
            df.append(correction_results)

        df = pd.DataFrame(df)
        return df

    # Visualize
    def show_lines(self, arrow_step=10, arrow_size=20):
        """ Display annotated shot lines on a 2d graph in CDP coordinates. """
        fig, ax = plt.subplots(figsize=(14, 8))

        depths = np.array([field.depth for field in self.fields])
        colors = ['black', 'firebrick', 'gold', 'limegreen', 'magenta'] * 5
        depth_to_color = dict(zip(sorted(np.unique(depths)), colors))

        # Data
        for i, field in enumerate(self.fields):
            color = depth_to_color[field.depth]
            values = field.geometry.headers[['CDP_X', 'CDP_Y']].values
            x, y = values[:, 0], values[:, 1]
            ax.plot(x, y, color)

            idx = x.size // 2
            ax.annotate('', size=arrow_size,
                        xytext=(x[idx-arrow_step], y[idx-arrow_step]),
                            xy=(x[idx+arrow_step], y[idx+arrow_step]),
                        arrowprops=dict(arrowstyle="->", color=color))

            ax.annotate(i, xy=(x[0], y[0]), size=12)

        # Annotations
        ax.set_title('2D profiles', fontsize=26)
        ax.set_xlabel('CDP_X', fontsize=22)
        ax.set_ylabel('CDP_Y', fontsize=22)
        ax.grid()
        fig.show()


    def show_bubblemap(self, savepath=None):
        """ Display annotated shot lines and their intersections on a 2d interactive graph in CDP coordinates. """
        fig = go.Figure()

        depths = np.array([field.depth for field in self.fields])
        colors = ['black', 'firebrick', 'gold', 'limegreen', 'magenta'] * 5
        depth_to_color = dict(zip(sorted(np.unique(depths)), colors))

        intersections_df = self.intersections_df()

        # Line for each SEG-Y
        for i, field in enumerate(self.fields):
            correction_results = field.correction_results
            values = field.geometry.headers[['CDP_X', 'CDP_Y']].values
            color_ = depth_to_color[field.depth]

            name_ = f'{i} : "{field.short_name}.sgy"'
            hovertemplate_ = (f' #{i} <br>'
                              f' FIELD : "{field.short_name}.sgy" <br>'
                              f' DEPTH : {field.depth} <br>'
                               ' CDP_X : %{x:,d} <br>'
                               ' CDP_Y : %{y:,d} <br>'
                               ' TSF : %{customdata} <br>'
                              f' MEAN INTERSECTION CORR : {correction_results["mean_corr_intersections"]:3.3f} <br>'
                              f' MEAN RECOMPUTED CORR : {correction_results["mean_recomputed_corr"]:3.3f}'
                               '<extra></extra>')

            step = 30
            fig.add_trace(go.Scatter(x=values[::step, 0], y=values[::step, 1],
                                     customdata=field.geometry.headers['TRACE_SEQUENCE_FILE'][::step],
                                     name=name_, hovertemplate=hovertemplate_,
                                     mode='lines',
                                     line=dict(color=color_, width=2)))

        # Markers on intersections
        for key, intersection in self.intersections.items():
            # Retrieve data
            i, j = key[:2]
            field_0, field_1 = intersection.field_0, intersection.field_1
            x, y = (intersection.coordinates_0 + intersection.coordinates_1) // 2

            matching_results = intersection.matching_results
            corr, shift = matching_results['corr'], matching_results['shift']
            angle, gain = matching_results['angle'], matching_results['gain']

            # HTML things
            name_ = f'"{field_0.short_name}.sgy" X "{field_1.short_name}.sgy"'
            hovertemplate_ = (f' ({i}, {j}) <br>'
                              f' {name_} <br>'
                               ' CDP_X : %{x:,d} <br>'
                               ' CDP_Y : %{y:,d} <br>'
                              f' BEST_CORR   : {corr:3.3f} <br>'
                              f' BEST_PCORR  : {(1 + corr)/2:3.3f} <br>'
                              f' SHIFT       : {shift:3.3f} <br>'
                              f' ANGLE       : {angle:3.3f} <br>'
                              f' GAIN        : {gain:3.3f} <extra></extra>')

            size_ = 7 + (1 - corr) * 10
            color_ = 'red' if intersections_df.loc[[key]]['suspicious'].all() else 'green'

            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     name=name_, hoverlabel={},
                                     hovertemplate=hovertemplate_,
                                     showlegend=False,
                                     marker=dict(size=size_, color=color_)))


        fig.update_layout(title=f'2D SEG-Y<br>{len(self.intersections)} intersections',
                          xaxis_title='CDP_X', yaxis_title='CDP_Y',
                          width=1200, height=500, margin=dict(l=10, r=10, t=40, b=10))
        fig.show()

        if savepath is not None:
            fig.write_html(savepath)

    def show_histogram(self, keys=('corr', 'shift', 'angle', 'gain'), **kwargs):
        """ Display histogram of mis-tie values across all intersections. """
        data = [self.get_matched_value(key) for key in keys]

        kwargs = {
            'title': list(keys),
            'xlabel': list(keys),
            'combine': 'separate',
            'ncols': 4,
            **kwargs
        }
        return plot(data, mode='histogram', **kwargs)



@njit
def distribute_misties(a, b, n, skip_index=-1, max_iters=100, alpha=0.75, tolerance=0.00001):
    """ Distribute misties `b` on intersections `a` over `n` fields by a iterative optimization procedure.
    Bishop, Nunns "`Correcting amplitude, time, and phase mis-ties in seismic data
    <https://www.researchgate.net/publication/249865260>`_"

    Probably, not as fast as highly-optimized linear solvers, but allows for more flexibility.
    Also, usual run time is less than 1ms, so it is fast enough anyways.

    Parameters
    ----------
    a : np.ndarray
        (M, 2)-shaped matrix that describes geometry of intersections.
        Each row is a pair of indices of intersecting lines.
    b : np.ndarray
        (M,)-shaped vector with misties on each intersection.
    n : int
        Number of lines in the intersections.
        For each of them, we compute a distributed mistie as a result of this function.
    """
    x = np.zeros(n)
    errors = np.empty(max_iters)

    for iteration in range(max_iters):
        # Stop condition: no further decrease in error
        xk, xl = x[a[:, 0]], x[a[:, 1]]
        errors[iteration] = ((b - (xk - xl)) ** 2).mean() ** (1 / 2)
        if iteration != 0:
            stop_condition = (errors[iteration - 1] - errors[iteration]) / errors[iteration - 1]
            if stop_condition < tolerance or errors[iteration] == 0.0:
                break

        # Compute next iteration of solution
        x_next = x.copy()

        for j in range(n):
            if j == skip_index:
                continue

            d, s = 0, 0.0 # number of intersections / sum of discrepancies
            for i, idx in enumerate(a):
                k, l = idx

                if k == j:
                    s += b[i] - (x[k] - x[l])
                    d += 1
                if l == j:
                    s += (x[k] - x[l]) - b[i]
                    d += 1

            if d != 0:
                x_next[j] += (alpha / d) * s

        x = x_next

    return x, errors[:iteration+1]
