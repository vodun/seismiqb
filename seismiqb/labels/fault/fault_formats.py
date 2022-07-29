""" Mixins to deal with fault storing files. """

import os
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .fault_postprocessing import split_array
from ...utils import CharismaMixin, make_interior_points_mask

class FaultSticksMixin(CharismaMixin):
    """ Mixin to load and dump FaultSticks. """
    FAULT_STICKS_SPEC = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    REDUCED_FAULT_STICKS_SPEC = ['iline', 'xline', 'height', 'name', 'number']

    @classmethod
    def read_df(cls, path):
        """ Create pandas.DataFrame from FaultSticks/CHARISMA file. """
        with open(path, encoding='utf-8') as file:
            line_len = len([item for item in file.readline().split(' ') if len(item) > 0])

        if line_len == 0:
            return pd.DataFrame({})

        if line_len == 3:
            names = cls.REDUCED_CHARISMA_SPEC
        elif line_len == 5:
            names = cls.REDUCED_FAULT_STICKS_SPEC
        elif line_len == 8:
            names = cls.FAULT_STICKS_SPEC
        elif line_len >= 9:
            names = cls.CHARISMA_SPEC
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        return pd.read_csv(path, sep=r'\s+', names=names)

    def df_to_sticks(self, df, return_direction=False):
        """ Transform initial pandas.DataFrame with sticks to array of sticks. """
        if len(df) == 0:
            raise ValueError('Empty DataFrame (possibly wrong coordinates).')
        col, direction = None, None

        ilines_diff = sum(df.iline[1:].values - df.iline[:-1].values == 0)
        xlines_diff = sum(df.xline[1:].values - df.xline[:-1].values == 0)
        if ilines_diff > xlines_diff: # Use iline as an index
            col = 'iline'
            direction = 0
        else: # Use xline as an index
            col = 'xline'
            direction = 1

        if 'number' in df.columns: # Dataframe has stick index
            col = 'number'

        if col is None:
            raise ValueError('Wrong format of sticks: there is no column to group points into sticks.')

        df = df.sort_values('height')
        sticks = df.groupby(col).apply(lambda x: x[self.COLUMNS].values).reset_index(drop=True)

        return (sticks, direction) if return_direction else sticks

    def remove_broken_sticks(self, sticks):
        """ Remove <<strange>> sticks. """
        # Remove sticks with horizontal parts.
        mask = sticks.apply(lambda x: len(np.unique(np.array(x)[:, 2])) == len(x))
        if not mask.all():
            warnings.warn(f'{self.name}: Fault has horizontal parts of sticks.')
        sticks = sticks.loc[mask]

        # Remove sticks with one node.
        mask = sticks.apply(len) > 1
        if not mask.all():
            warnings.warn(f'{self.name}: Fault has one-point sticks.')
        sticks = sticks.loc[mask]

        # Filter faults with one stick.
        if len(sticks) == 1:
            warnings.warn(f'{self.name}: Fault has an only one stick')
            sticks = pd.Series()
        elif len(sticks) == 0:
            warnings.warn(f'{self.name}: Empty file')
            sticks = pd.Series()

        return sticks

    def load_fault_sticks(self, path, transform=True, verify=True,
                          recover_lines=True, remove_broken_sticks=False, **kwargs):
        """ Get sticks from csv file. """
        df = self.read_df(path)

        if len(df) == 0:
            self._sticks = [[]]
            self.direction = 1
            return

        if recover_lines and 'cdp_x' in df.columns:
            df = self.recover_lines_from_cdp(df)

        points = df[self.REDUCED_CHARISMA_SPEC].values

        if transform:
            points = self.field_reference.geometry.lines_to_cubic(points)
        df[self.REDUCED_CHARISMA_SPEC] = np.round(points).astype(np.int32)

        if verify:
            mask = make_interior_points_mask(points, self.field_reference.shape)
            df = df.iloc[mask]

        sticks, direction = self.df_to_sticks(df, return_direction=True)
        if remove_broken_sticks:
            sticks = self.remove_broken_sticks(sticks)

        # Order sticks with respect of fault direction. Is necessary to perform following triangulation.
        if len(sticks) > 1:
            pca = PCA(1)
            coords = pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values]))
            indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
            sticks = sticks.iloc[indices]

        self._sticks = sticks.values

        # fix several slides sticks
        if direction is not None:
            ptp = np.array([np.ptp(stick[:, direction]) for stick in self.sticks])
            if (ptp > 2).any():
                warnings.warn(f"{self.path}: there sticks on several slides in both directions")

            for stick in self.sticks[np.logical_and(ptp > 0, ptp <= 2)]:
                stick[:, direction] = stick[0, direction]

        self.direction = direction

    def dump_fault_sticks(self, path, sticks_step=10, stick_nodes_step=10):
        """ Dump fault sticks. """
        path = self.field.make_path(path, name=self.field.short_name, makedirs=False)

        sticks_df = []
        for stick_idx, stick in enumerate(self.sticks):
            stick = self.field.geometry.cubic_to_lines(stick).astype(int)
            cdp = self.field.geometry.lines_to_cdp(stick[:, :2])
            df = {
                'INLINE-': 'INLINE-',
                'iline': stick[:, 0],
                'xline': stick[:, 1],
                'cdp_x': cdp[:, 0],
                'cdp_y': cdp[:, 1],
                'height': stick[:, 2],
                'name': os.path.basename(path),
                'number': stick_idx
            }
            sticks_df.append(pd.DataFrame(df))
        sticks_df = pd.concat(sticks_df)
        sticks_df.to_csv(path, header=False, index=False, sep=' ')

    def show_file(self):
        """ Show initial FaultSticks file. """
        with open(self.path, encoding='utf-8') as f:
            print(f.read())

    @classmethod
    def check_format(cls, path, verbose=False):
        """ Find errors in fault file.

        Parameters
        ----------
        path : str
            path to file or glob expression
        verbose : bool
            response if file is succesfully readed.
        """
        for filename in glob.glob(path):
            if os.path.splitext(filename)[1] == '.dvc':
                continue
            try:
                df = cls.read_df(filename)
                sticks = cls.df_to_sticks(cls, df)
            except ValueError:
                print(filename, ': wrong format')
            else:
                if 'name' in df.columns and len(df.name.unique()) > 1:
                    print(filename, ': file must be splitted.')
                    continue

                if len(sticks) == 1:
                    print(filename, ': fault has an only one stick')
                    continue

                if any(len(item) == 1 for item in sticks):
                    print(filename, ': fault has one point stick')
                    continue
                mask = sticks.apply(lambda x: len(np.unique(np.array(x)[:, 2])) == len(x))
                if not mask.all():
                    print(filename, ': fault has horizontal parts of sticks.')
                    continue

                if verbose:
                    print(filename, ': OK')

    @classmethod
    def split_file(cls, path, dst):
        """ Split file with multiple faults into separate files. """
        if dst and not os.path.isdir(dst):
            os.makedirs(dst)
        df = pd.read_csv(path, sep=r'\s+', names=cls.FAULT_STICKS)
        df.groupby('name').apply(cls.fault_to_csv, dst=dst)

    @classmethod
    def fault_to_csv(cls, df, dst):
        """ Save the fault to csv. """
        df.to_csv(os.path.join(dst, df.name), sep=' ', header=False, index=False)


class FaultSerializationMixin:
    """ Mixin for npy/npz storage of fault components (points, sticks, nodes, simplices). """
    def load_npz(self, path):
        """ Load fault points, nodes and sticks from npz file. """
        npzfile = np.load(path, allow_pickle=False)

        sticks = npzfile.get('sticks')
        sticks_labels = npzfile.get('sticks_labels')

        self.from_objects({
            'points': npzfile['points'],
            'nodes': npzfile.get('nodes'),
            'simplices': npzfile.get('simplices'),
            'sticks': self.labeled_array_to_sticks(sticks, sticks_labels),
        })

        self.direction = npzfile.get('direction')

    def load_npy(self, path):
        """ Load fault points from npy file. """
        points = np.load(path, allow_pickle=False)
        self._points = points

    def dump_npz(self, path):
        """ Dump fault to npz. """
        path = self.field.make_path(path, name=self.short_name, makedirs=False)

        if self.has_component('sticks'):
            sticks, sticks_labels = self.sticks_to_labeled_array(self.sticks)
        else:
            sticks, sticks_labels = np.zeros((0, 3)), np.zeros((0, 1))

        np.savez(path, points=self._points, nodes=self._nodes, simplices=self._simplices,
                 sticks=sticks, sticks_labels=sticks_labels, direction=self.direction)


    def sticks_to_labeled_array(self, sticks):
        """ Auxilary method to dump fault into npz with allow_pickle=False. """
        labels = sum([[i] * len(item) for i, item in enumerate(sticks)], [])
        return np.concatenate(sticks), labels

    def labeled_array_to_sticks(self, sticks, labels):
        """ Auxilary method to dump fault into npz with allow_pickle=False. """
        sticks = split_array(sticks, labels)
        array = np.empty(len(sticks), dtype=object)
        for i, item in enumerate(sticks):
            array[i] = item
        return array
