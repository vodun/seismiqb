""" Accumulator for 3d volumes. """
import os

import h5py
import hdf5plugin
import numpy as np
from sklearn.linear_model import LinearRegression
from multiprocess import Process, JoinableQueue

from batchflow import Notifier

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
    def __init__(self, shape=None, origin=None, orientation=0, dtype=np.float32, transform=None, path=None,
                 dataset_kwargs=None, **kwargs):
        # Dimensionality and location, corrected on `orientation`
        self.orientation = orientation
        self.shape = self.reorder(shape)
        self.origin = self.reorder(origin)
        self.location = self.reorder([slice(start, start + shape)
                                      for start, shape in zip(self.origin, self.shape)])

        # Properties of storages
        self.dtype = dtype
        self.transform = transform if transform is not None else lambda array: array

        # Container definition
        if path is not None:
            if isinstance(path, str) and os.path.exists(path):
                os.remove(path)
            self.path = path

            self.file = h5py.File(path, mode='w-')
            self.dataset_kwargs = dataset_kwargs or {}

        self.type = os.path.splitext(path)[1][1:] if path is not None else 'numpy'

        self.aggregated = False
        self.kwargs = kwargs

    def reorder(self, sequence):
        """ Reorder `sequence` with the `orientation` of accumulator. """
        if self.orientation == 1:
            sequence = np.array([sequence[1], sequence[0], sequence[2]])
        return sequence

    # Placeholder management
    def create_placeholder(self, name=None, dtype=None, fill_value=None):
        """ Create named storage as a dataset of HDF5 or plain array. """
        if self.type in ['hdf5', 'qhdf5']:
            placeholder = self.file.create_dataset(name, shape=self.shape, dtype=dtype,
                                                   fillvalue=fill_value, **self.dataset_kwargs)
        elif self.type == 'numpy':
            placeholder = np.full(shape=self.shape, fill_value=fill_value, dtype=dtype)

        setattr(self, name, placeholder)

    def remove_placeholder(self, name=None):
        """ Remove created placeholder. """
        if self.type in ['hdf5', 'qhdf5']:
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

        # Correct orientation
        location = self.reorder(location)
        if self.orientation == 1:
            crop = crop.transpose(1, 0, 2)

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
        # TODO: open resulting HDF5 file with `Geometry` and return it instead?
        self.aggregated = True
        if self.type in ['hdf5', 'qhdf5']:
            projection_name = 'projection_i' if self.orientation == 0 else 'projection_x'
            self.file[projection_name] = self.file['data']
            self.file.close()
            self.file = h5py.File(self.path, 'r+')
            self.data = self.file['data']
        else:
            if self.orientation == 1:
                self.data = self.data.transpose(1, 0, 2)
        return self.data

    def _aggregate(self):
        """ Aggregate placeholders into resulting array. Changes `data` placeholder inplace. """
        raise NotImplementedError

    def clear(self):
        """ Remove placeholders from memory and disk. """
        if self.type in ['hdf5', 'qhdf5']:
            os.remove(self.path)

    @property
    def result(self):
        """ Reference to the aggregated result. """
        if not self.aggregated:
            self.aggregate()
        return self.data

    def export_to_hdf5(self, path=None, projections=(0,), pbar='t', dtype=None, transform=None, dataset_kwargs=None):
        """ Export `data` attribute to a file. """
        if self.type != 'numpy' or self.orientation != 0:
            raise NotImplementedError('`export_to_hdf5` works only with `numpy` accumulators with `orientation=0`!')

        # Parse parameters
        from ..geometry.conversion_mixin import ConversionMixin #pylint: disable=import-outside-toplevel
        if isinstance(path, str) and os.path.exists(path):
            os.remove(path)

        dtype = dtype or self.dtype
        transform = transform or (lambda array: array)
        dataset_kwargs = dataset_kwargs or dict(hdf5plugin.Blosc(cname='lz4hc', clevel=6, shuffle=0))

        data = self.data

        with h5py.File(path, mode='w-') as file:
            with Notifier(pbar, total=sum(data.shape[axis] for axis in projections)) as progress_bar:
                for axis in projections:
                    projection_name = ConversionMixin.PROJECTION_NAMES[axis]
                    projection_transposition = ConversionMixin.TO_PROJECTION_TRANSPOSITION[axis]
                    projection_shape = np.array(data.shape)[projection_transposition]

                    dataset_kwargs_ = {'chunks': (1, *projection_shape[1:]), **dataset_kwargs}
                    projection = file.create_dataset(projection_name, shape=projection_shape, dtype=self.dtype,
                                                    **dataset_kwargs_)

                    for i in range(data.shape[axis]):
                        projection[i] = transform(np.take(data, i, axis=axis))
                        progress_bar.update()
        return h5py.File(path, mode='r')

    # Pre-defined transforms
    @staticmethod
    def prediction_to_int8(array):
        """ Convert a float array with values in [0.0, 1.0] to an int8 array with values in [-128, +127]. """
        array *= 255
        array -= 128
        return array.astype(np.int8)

    @staticmethod
    def int8_to_prediction(array):
        """ Convert an int8 array with values in [-128, +127] to a float array with values in [0.0, 1.0]. """
        array = array.astype(np.float32)
        array += 128
        array /= 255
        return array

    @staticmethod
    def prediction_to_uint8(array):
        """ Convert a float array with values in [0.0, 1.0] to an int8 array with values in [0, 255]. """
        array *= 255
        return array.astype(np.uint8)

    @staticmethod
    def uint8_to_prediction(array):
        """ Convert an int8 array with values in [0, 255] to a float array with values in [0.0, 1.0]. """
        array = array.astype(np.float32)
        array /= 255
        return array

    # Alternative constructors
    @classmethod
    def from_aggregation(cls, aggregation='max', shape=None, origin=None, dtype=np.float32, fill_value=None,
                         transform=None, path=None, dataset_kwargs=None, **kwargs):
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
                                                 transform=transform, path=path,
                                                 dataset_kwargs=dataset_kwargs, **kwargs)

    @classmethod
    def from_grid(cls, grid, aggregation='max', dtype=np.float32, fill_value=None, transform=None, path=None,
                  dataset_kwargs=None, **kwargs):
        """ Infer necessary parameters for accumulator creation from a passed grid. """
        return cls.from_aggregation(aggregation=aggregation, dtype=dtype, fill_value=fill_value,
                                    shape=grid.shape, origin=grid.origin, orientation=grid.orientation,
                                    transform=transform, path=path, dataset_kwargs=dataset_kwargs, **kwargs)


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
        if dtype == np.int8:
            raise NotImplementedError('`mean` accumulation is unavailable for `dtype=in8`. Use `weighted` aggregation.')
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

    NOTE: add support of weights incoming along with a data-crop.
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


class ChunksAccumulator:
    """ Accumulator which aggregates whole slices in memory and then write it into file. Can aggregate
    data from several sources (e.g., predictions from different models).

    Parameters
    ----------
    args : sequence
        Args for Accumulator3D.from_aggregation.
    shape : sequence
        Shape of the placeholder.
    n_slides : int
        The number of slides in chunk to aggregate in memory.
    n_sorces : int
        The number of sources of data.
    orientation : 0 or 1
        Orientation of predictions.
    postprocessing : callable or None
        Function to process chunk with the following signature:

            Parameters
            ----------
            chunk : np.ndarray
                Chunk to process.
            origin : tuple
                Origin of chunk as tuple of 3 coordinates.
            shape : tuple
                Shape of the whole accumulator.
            prev_chunk : np.ndarray
                Previous chunk to use if processing use some kind of padding.
            global_origin : tuple
                Origin of the whole accumulator.

            Return
            ------
            (np.ndarray, tuple)
                chunk :
                    Processed chunk, possibly with a different origin (e.g., if processing includes convolutions).
                origin :
                    Origin of the processed chunk.

    kwargs : dict
        Kwargs for  Accumulator3D.from_aggregation.
    """
    def __init__(self, *args, shape=None, n_slides=40, n_sources=1, orientation=0, postprocessing=None,
                 origin=(0, 0, 0), **kwargs):
        self.args, self.kwargs = args, kwargs
        self.origin = tuple(origin)
        self.orientation = orientation
        self.n_slides = n_slides
        self.n_sources = n_sources
        self.shape = shape
        self._processes = []

        self.chunks_queue = JoinableQueue(maxsize=3)
        self.postprocessing = postprocessing

        self.accumulator_process = Process(target=self.collect_chunks, args=(self.chunks_queue, ))
        self.accumulator_process.start()

        self.chunk_accumulators = [{} for _ in range(n_sources)]

    def update(self, crop, location, source_idx=0):
        """ Update underlying storages in supplied `location` with data from `crop`. """
        origin = location[self.orientation].start - location[self.orientation].start % self.n_slides
        if origin not in self.chunk_accumulators[source_idx]:
            self.aggregate_last(source_idx) # TODO: fix for prefetch
            self.create_chunk(origin, source_idx)
        self.chunk_accumulators[source_idx][origin].update(crop, location)

    def create_chunk(self, origin, source_idx):
        """ Create accumulator for chunk in memory. """
        shape = np.array(self.shape)
        shape[self.orientation] = self.n_slides
        shape = tuple(shape)

        origin_ = list(self.origin)
        origin_[self.orientation] = origin
        origin_ = tuple(origin_)

        kwargs = {**self.kwargs, 'shape': shape, 'origin': origin_, 'path': None}

        self.chunk_accumulators[source_idx][origin] = Accumulator3D.from_aggregation(*self.args, **kwargs)

    def aggregate_last(self, source_idx):
        """ Aggregate last ready chunk. """
        if len(self.chunk_accumulators[source_idx]) > 0:
            origin = list(self.chunk_accumulators[source_idx].keys())[-1]
            chunk = self.chunk_accumulators[source_idx].pop(origin)
            self.chunks_queue.put((origin, chunk))

    def aggregate(self):
        """ Aggregate whole accumulator. """
        for source_idx in range(self.n_sources):
            self.aggregate_last(source_idx)

        self.chunks_queue.put((None, None))
        self.accumulator_process.join()
        self.file = h5py.File(self.kwargs['path'], 'r+')
        self.data = self.file['data']
        return self.data

    def collect_chunks(self, chunks_queue):
        """ Process chunks and put them into chunks accumulator. """
        # TODO: last lines when range is not None
        accumulator = Accumulator3D.from_aggregation(*self.args, shape=self.shape, orientation=self.orientation,
                                                     origin=self.origin, **self.kwargs)
        global_origin = self.origin
        origin_ = list(global_origin)

        origin, chunk = chunks_queue.get()
        prev_chunk = None
        while chunk is not None:
            chunk = chunk.aggregate()
            if self.postprocessing:
                origin_[self.orientation] = origin
                chunk, origin, prev_chunk = (
                    *self.postprocessing(chunk, origin_, self.shape, prev_chunk, global_origin),
                    chunk
                )
            location = [slice(loc, loc+length) for loc, length in zip(global_origin, self.shape)]
            location[self.orientation] = slice(origin, origin+chunk.shape[self.orientation])
            accumulator.update(chunk, location)
            origin, chunk = chunks_queue.get()
        accumulator.aggregate()
