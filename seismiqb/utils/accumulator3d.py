""" Accumulator for 3d volumes. """
import os

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression

from .functions import triangular_weights_function_nd



class Accumulator3D:
    """ Base class to aggregate predicted sub-volumes into a larger 3D cube.
    Can accumulate data in memory (Numpy arrays) or on disk (HDF5 datasets).

    Type of aggregation is defined in subclasses, that must implement `__init__`, `_update` and `_aggregate` methods.
    The main result in subclasses should be stored in `data` attribute, which is accessed by the base class.

    Supposed to be used in combination with `:class:.~RegularGrid` and
    `:meth:.~SeismicCropBatch.update_accumulator` in a following manner:
        - `RegularGrid` defines how to split desired cube range into small crops
        - `Accumulator3D` creates necessary placeholders for a desired type of aggregation
        - `update_accumulator` action of pipeline passes individual crops (and their locations) to
        update those placeholders (see `:meth:~.update`)
        - `:meth:~.aggregate` is used to get the resulting volume
        - `:meth:~.clear` can be optionally used to remove array references and HDF5 file from disk

    This class is an alternative to `:meth:.~SeismicDataset.assemble_crops`, but allows to
    greatly reduce memory footprint of crop aggregation by up to `overlap_factor` times.
    Also, as this class updates rely on `location`s of crops, it can take crops in any order.

    Note that not all pixels of placeholders will be updated with data due to removal of dead traces,
    so we have to be careful with initialization!

    Parameters
    ----------
    shape : sequence
        Shape of the placeholder.
    origin : sequence
        The upper left point of the volume: used to shift crop's locations.
    dtype : np.dtype
        Dtype of storage. Must be either integer or float.
    transform : callable, optional
        Additional function to call before storing the crop data.
    path : str or file-like object, optional
        If provided, then we use HDF5 datasets instead of regular Numpy arrays, storing the data directly on disk.
        After the initialization, we keep the file handle in `w-` mode during the update phase.
        After aggregation, we re-open the file to automatically repack it in `r` mode.
    kwargs : dict
        Other parameters are passed to HDF5 dataset creation.
    """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None, **kwargs):
        # Dimensionality and location
        self.shape = shape
        self.origin = origin
        self.location = [slice(start, start + shape) for start, shape in zip(self.origin, self.shape)]

        # Properties of storages
        self.dtype = dtype
        self.transform = transform if transform is not None else lambda array: array

        # Container definition
        if path is not None:
            if isinstance(path, str) and os.path.exists(path):
                os.remove(path)
            self.path = path

            self.file = h5py.File(path, mode='w-')
        self.type = os.path.splitext(path)[1][1:] if path is not None else 'numpy'

        self.aggregated = False
        self.kwargs = kwargs

    # Placeholder management
    def create_placeholder(self, name=None, dtype=None, fill_value=None):
        """ Create named storage as a dataset of HDF5 or plain array. """
        if self.type in ('hdf5', 'blosc'):
            placeholder = self.file.create_dataset(name, shape=self.shape, dtype=dtype, fillvalue=fill_value)
        elif self.type == 'numpy':
            placeholder = np.full(shape=self.shape, fill_value=fill_value, dtype=dtype)

        setattr(self, name, placeholder)

    def remove_placeholder(self, name=None):
        """ Remove created placeholder. """
        if self.type in ['hdf5', 'blosc']:
            del self.file[name]
        setattr(self, name, None)


    def update(self, crop, location):
        """ Update underlying storages in supplied `location` with data from `crop`. """
        if self.aggregated:
            raise RuntimeError('Aggregated data has been already computed!')

        # Check all shapes for compatibility
        for s, slc in zip(crop.shape, location):
            if slc.step and slc.step != 1:
                raise ValueError(f"Invalid step in location {location}")

            if s < slc.stop - slc.start:
                raise ValueError(f"Inconsistent crop_shape {crop.shape} and location {location}")

        # Compute correct shapes
        loc, loc_crop = [], []
        for xmin, slc, xmax in zip(self.origin, location, self.shape):
            loc.append(slice(max(0, slc.start - xmin), min(xmax, slc.stop - xmin)))
            loc_crop.append(slice(max(0, xmin - slc.start), min(xmax + xmin - slc.start , slc.stop - slc.start)))

        # Actual update
        crop = self.transform(crop[tuple(loc_crop)])
        location = tuple(loc)
        self._update(crop, location)

    def _update(self, crop, location):
        """ Update placeholders with data from `crop` at `locations`. """
        _ = crop, location
        raise NotImplementedError

    def aggregate(self):
        """ Finalize underlying storages to create required aggregation. """
        if self.aggregated:
            raise RuntimeError('All data in the container has already been cleared!')
        self._aggregate()

        # Re-open the HDF5 file to force flush changes and release disk space from deleted datasets
        # Also add alias to `data` dataset, so the resulting cube can be opened by `Geometry`
        if self.type == 'hdf5':
            self.file['projection_i'] = self.file['data']
            self.file.close()
            self.file = h5py.File(self.path, 'r+')
            self.data = self.file['data']

        self.aggregated = True
        return self.data

    def _aggregate(self):
        """ Aggregate placeholders into resulting array. Changes `data` placeholder inplace. """
        raise NotImplementedError

    def clear(self):
        """ Remove placeholders from memory and disk. """
        if self.type in ['hdf5', 'blosc']:
            os.remove(self.path)

    @property
    def result(self):
        """ Reference to the aggregated result. """
        if not self.aggregated:
            self.aggregate()
        return self.data

    @classmethod
    def from_aggregation(cls, aggregation='max', shape=None, origin=None, dtype=np.float32, fill_value=None,
                         transform=None, path=None, **kwargs):
        """ Initialize chosen type of accumulator aggregation. """
        class_to_aggregation = {
            MaxAccumulator3D: ['max', 'maximum'],
            MeanAccumulator3D: ['mean', 'avg', 'average'],
            GMeanAccumulator3D: ['gmean', 'geometric'],
            WeightedSumAccumulator3D: ['weighted'],
            ModeAccumulator3D: ['mode']
        }
        aggregation_to_class = {alias: class_ for class_, lst in class_to_aggregation.items()
                                for alias in lst}

        return aggregation_to_class[aggregation](shape=shape, origin=origin, dtype=dtype, fill_value=fill_value,
                                                 transform=transform, path=path, **kwargs)

    @classmethod
    def from_grid(cls, grid, aggregation='max', dtype=np.float32, fill_value=None, transform=None, path=None, **kwargs):
        """ Infer necessary parameters for accumulator creation from a passed grid. """
        return cls.from_aggregation(aggregation=aggregation, dtype=dtype, fill_value=fill_value,
                                    shape=grid.shape, origin=grid.origin, orientation=grid.orientation,
                                    transform=transform, path=path, **kwargs)


class MaxAccumulator3D(Accumulator3D):
    """ Accumulator that takes maximum value of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, fill_value=None, transform=None, path=None, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        min_value = np.finfo(dtype).min if 'float' in dtype.__name__ else np.iinfo(dtype).min
        self.fill_value = fill_value if fill_value is not None else min_value
        self.create_placeholder(name='data', dtype=self.dtype, fill_value=self.fill_value)

    def _update(self, crop, location):
        self.data[location] = np.maximum(crop, self.data[location])

    def _aggregate(self):
        pass


class MeanAccumulator3D(Accumulator3D):
    """ Accumulator that takes mean value of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=0)
        self.create_placeholder(name='counts', dtype=np.int8, fill_value=0)

    def _update(self, crop, location):
        self.data[location] += crop
        self.counts[location] += 1

    def _aggregate(self):
        #pylint: disable=access-member-before-definition
        if self.type == 'hdf5':
            # Amortized updates for HDF5
            for i in range(self.data.shape[0]):
                counts = self.counts[i]
                counts[counts == 0] = 1
                if np.issubdtype(self.dtype, np.floating):
                    self.data[i] /= counts
                else:
                    self.data[i] //= counts

        elif self.type == 'numpy':
            self.counts[self.counts == 0] = 1
            if np.issubdtype(self.dtype, np.floating):
                self.data /= self.counts
            else:
                self.data //= self.counts

        # Cleanup
        self.remove_placeholder('counts')


class GMeanAccumulator3D(Accumulator3D):
    """ Accumulator that takes geometric mean value of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=1)
        self.create_placeholder(name='counts', dtype=np.int8, fill_value=0)

    def _update(self, crop, location):
        self.data[location] *= crop
        self.counts[location] += 1

    def _aggregate(self):
        #pylint: disable=access-member-before-definition
        if self.type == 'hdf5':
            # Amortized updates for HDF5
            for i in range(self.data.shape[0]):
                counts = self.counts[i]
                counts[counts == 0] = 1

                counts = counts.astype(np.float32)
                counts **= -1
                self.data[i] **= counts

        elif self.type == 'numpy':
            self.counts[self.counts == 0] = 1

            self.counts = self.counts.astype(np.float32)
            self.counts **= -1
            self.data **= self.counts

        # Cleanup
        self.remove_placeholder('counts')


class ModeAccumulator3D(Accumulator3D):
    """ Accumulator that takes mode value in overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32,
                 n_classes=2, transform=None, path=None, **kwargs):
        # Create placeholder with counters for each class
        self.fill_value = 0
        self.n_classes = n_classes

        shape = (*shape, n_classes)
        origin = (*origin, 0)

        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=self.fill_value)

    def _update(self, crop, location):
        # Update class counters in location
        crop = np.eye(self.n_classes)[crop]
        self.data[location] += crop

    def _aggregate(self):
        # Choose the most frequently seen class value
        if self.type == 'hdf5':
            for i in range(self.data.shape[0]):
                self.data[i] = np.argmax(self.data[i], axis=-1)

        elif self.type == 'numpy':
            self.data = np.argmax(self.data, axis=-1)

class WeightedSumAccumulator3D(Accumulator3D):
    """ Accumulator that takes weighted sum of overlapping crops. Accepts `weights_function`
    for making weights for each crop into the initialization.

    NOTE: add later support of
    (i) weights incoming along with a data-crop.
    """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None,
                 weights_function=triangular_weights_function_nd, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=0)
        self.create_placeholder(name='weights', dtype=np.float32, fill_value=0)
        self.weights_function = weights_function

    def _update(self, crop, location):
        # Weights matrix for the incoming crop
        crop_weights = self.weights_function(crop)
        self.data[location] = ((crop_weights * crop + self.data[location] * self.weights[location]) /
                               (crop_weights + self.weights[location]))
        self.weights[location] += crop_weights

    def _aggregate(self):
        # Cleanup
        self.remove_placeholder('weights')

class RegressionAccumulator(Accumulator3D):
    """ Accumulator that fits least-squares regression to scale values of
    each incoming crop to match values of the overlap. In doing so, ignores nan-values.
    For aggregation uses weighted sum of crops. Weights-making for crops is controlled by
    `weights_function`-parameter.

    Parameters
    ----------
    shape : sequence
        Shape of the placeholder.
    origin : sequence
        The upper left point of the volume: used to shift crop's locations.
    dtype : np.dtype
        Dtype of storage. Must be either integer or float.
    transform : callable, optional
        Additional function to call before storing the crop data.
    path : str or file-like object, optional
        If provided, then we use HDF5 datasets instead of regular Numpy arrays, storing the data directly on disk.
        After the initialization, we keep the file handle in `w-` mode during the update phase.
        After aggregation, we re-open the file to automatically repack it in `r` mode.
    weights_function : callable
        Function that accepts a crop and returns matrix with weights of the same shape. Default scheme
        involves using larger weights in the crop-centre and lesser weights closer to the crop borders.
    rsquared_lower_bound : float
        Can be a number between 0 and 1 or `None`. If set to `None`, we use each incoming crop with
        predictions to update the assembled array. Otherwise, we use only those crops, that fit already
        filled data well enough, requiring r-squared of linear regression to be larger than the supplied
        parameter.
    regression_target : str
        Can be either 'assembled' (same as 'accumulated') or 'crop' (same as 'incoming'). If set to
        'assembled', the regression considers new crop as a regressor and already filled overlap as a target.
        If set to 'crop', incoming crop is the target in the regression. The choice of 'assembled'
        should yield more stable results.

    NOTE: As of now, relies on the order in which crops with data arrive. When the order of
    supplied crops is different, the result of aggregation might differ as well.
    """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None,
                 weights_function=triangular_weights_function_nd, rsquared_lower_bound=.2,
                 regression_target='assembled', **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        # Fill both placeholders with nans: in order to fit the regression
        # it is important to understand what overlap values are already filled.
        # NOTE: perhaps rethink and make weighted regression.
        self.create_placeholder(name='data', dtype=self.dtype, fill_value=np.nan)
        self.create_placeholder(name='weights', dtype=np.float32, fill_value=np.nan)

        self.weights_function = weights_function
        self.rsquared_lower_bound = rsquared_lower_bound or -1

        if regression_target in ('assembled', 'accumulated'):
            self.regression_target = 'assembled'
        elif regression_target in ('crop', 'incoming'):
            self.regression_target = 'crop'
        else:
            raise ValueError(f'Unknown regression target {regression_target}.')

    def _update(self, crop, location):
        # Scale incoming crop to better fit already filled data.
        # Fit is done via least-squares regression.
        overlap_data = self.data[location]
        overlap_weights = self.weights[location]
        crop_weights = self.weights_function(crop)

        # If some of the values are already filled, use regression to fit new crop
        # to what's filled.
        overlap_indices = np.where((~np.isnan(overlap_data)) & (~np.isnan(crop)))
        new_indices = np.where(np.isnan(overlap_data))

        if len(overlap_indices[0]) > 0:
            # Take overlap values from data-placeholder and the crop.
            # Select regression/target according to supplied parameter `regression_target`.
            if self.regression_target == 'assembled':
                xs, ys = crop[overlap_indices], overlap_data[overlap_indices]
            else:
                xs, ys = overlap_data[overlap_indices], crop[overlap_indices]

            # Fit new crop to already existing data and transform the crop.
            model = LinearRegression()
            model.fit(xs.reshape(-1, 1), ys.reshape(-1))

            # Calculating the r-squared of the fitted regression.
            a, b = model.coef_[0], model.intercept_
            xs, ys = xs.reshape(-1), ys.reshape(-1)
            rsquared = 1 - ((a * xs + b - ys) ** 2).mean() / ((ys - ys.mean()) ** 2).mean()

            # If the fit is bad (r-squared is too small), ignore the incoming crop.
            # If it is of acceptable quality, use it to update the assembled-array.
            if rsquared > self.rsquared_lower_bound:
                if self.regression_target == 'assembled':
                    crop = a * crop + b
                else:
                    crop = (crop - b) / a

                # Update location-slice with weighted average.
                overlap_data[overlap_indices] = ((overlap_weights[overlap_indices] * overlap_data[overlap_indices]
                                                + crop_weights[overlap_indices] * crop[overlap_indices]) /
                                                (overlap_weights[overlap_indices] + crop_weights[overlap_indices]))

                # Update weights over overlap.
                overlap_weights[overlap_indices] += crop_weights[overlap_indices]

                # Use values from crop to update the region covered by the crop and not yet filled.
                self.data[location][new_indices] = crop[new_indices]
                self.weights[location][new_indices] = crop_weights[new_indices]
        else:
            self.data[location] = crop
            self.weights[location] = crop_weights

    def _aggregate(self):
        # Clean-up
        self.remove_placeholder('weights')


class AccumulatorBlosc(Accumulator3D):
    """ Accumulate predictions into `BLOSC` file.
    Each of the saved slides supposed to be finalized, e.g. coming from another accumulator.
    During the aggregation, we repack the file to remove duplicates.

    Parameters
    ----------
    path : str
        Path to save `BLOSC` file to.
    orientation : int
        If 0, then predictions are stored as `projection_i` dataset inside the file.
        If 1, then predictions are stored as `projection_x` dataset inside the file and transposed before storing.
    aggregation : str
        Type of aggregation for duplicate slides.
        If `max`, then we take element-wise maximum.
        If `mean`, then take mean value.
        If None, then we take random slide.
    """
    def __init__(self, path, orientation=0, aggregation='max',
                 shape=None, origin=None, dtype=np.float32, transform=None, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=None)
        if orientation == 2:
            raise ValueError("Can't use BLOSC accumulator for a joined grid with mixed orientations!")

        self.type = 'blosc'
        self.path = path
        self.orientation = orientation
        self.aggregation = aggregation

        # Manage the `BLOSC` file
        from .geometry import BloscFile #pylint: disable=import-outside-toplevel
        self.file = BloscFile(path, mode='w')
        if orientation == 0:
            name = 'projection_i'
        elif orientation == 1:
            name = 'projection_x'
            shape = np.array(shape)[[1, 0, 2]]
        self.file.create_dataset(name, shape=shape, dtype=dtype)


    def _update(self, crop, location):
        crop = crop.astype(self.dtype)
        iterator = range(location[self.orientation].start, location[self.orientation].stop)

        # `i` is `loc_idx` shifted by `origin`
        for i, loc_idx in enumerate(iterator):
            slc = [slice(None), slice(None), slice(None)]
            slc[self.orientation] = i
            slide = crop[tuple(slc)]
            self.data[loc_idx, :, :] = slide.T if self.orientation == 1 else slide

    def _aggregate(self):
        self.file = self.file.repack(aggregation=self.aggregation)


    @classmethod
    def from_grid(cls, grid, aggregation='max', dtype=np.float32, transform=None, path=None, **kwargs):
        """ Infer necessary parameters for accumulator creation from a passed grid. """
        return cls(path=path, aggregation=aggregation, dtype=dtype, transform=transform,
                   shape=grid.shape, origin=grid.origin, orientation=grid.orientation, **kwargs)
