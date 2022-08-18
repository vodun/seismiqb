"""Maps and metrics for denoising seismic data."""
import numpy as np
from scipy import fftpack
import torch

from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio, mean_squared_error,\
                                    error_relative_global_dimensionless_synthesis, universal_image_quality_index

class DenoisingMetrics:
    """Class for metrics and color maps assotiated with denoising performance estimation."""
    @classmethod
    def get_metrics(cls, images, predictions, config=None):
        """Compute metrics related to denoising performance for a batch of images.

        Parameters
        ----------
        images : sequence
        predictions: sequence
        config : dict or None
            Specifies metrics to compute and their parameters.
            If None, computes [`structural_similarity_index_measure`, `peak_signal_noise_ratio`, `mean_squared_error`,
            `error_relative_global_dimensionless_synthesis`, `universal_image_quality_index`]
            from `torchmetrics.functional` with default arguments.
            Keys:
                metric_names : sequence
                    Sequence of metric names from {'ssim', 'psnr', 'ergas', 'uqi', 'mse'}.
                `metric_name` : dict, optional
                    Parameters for the corresponding function from `torchmetrics.functional`.

        Returns
        -------
        dict
            Dictionary with `metric_names` as keys and computed metrics as values.
        """
        metrics = {'ssim': structural_similarity_index_measure,
                   'psnr': peak_signal_noise_ratio,
                   'ergas': error_relative_global_dimensionless_synthesis,
                   'uqi': universal_image_quality_index,
                   'mse': mean_squared_error}

        metric_names = metrics.keys() if config is None else config['metric_names']
        if not np.all([metric in metrics for metric in metric_names]):
            raise ValueError

        config = config if config is not None else {}
        images = images if isinstance(images, torch.Tensor) else torch.tensor(images)
        predictions = predictions if isinstance(predictions, torch.Tensor) else torch.tensor(predictions)

        returns = {}
        for metric in metric_names:
            config[metric] = {} if metric not in config else config[metric]
            returns[metric] = metrics[metric](images, predictions, **config[metric]).item()
        return returns

    @classmethod
    def get_maps(cls, image, prediction, map_to='pred', config=None):
        """Compute color maps related to denoising performance for an image sample.

        Parameters
        ----------
        image : np.ndarray
        prediction : np.ndarray
        map_to : str
            If 'pred', statistics are computed between estimated noise and predicted image.
            If 'image', statistics are computed between estimated noise and source image.
            By default equals to 'pred'.
        config : dict or None
            Specifies maps to compute and their parameters.
            If None, computes [`local_similarity_map`, `local_correlation_map`, `fourier_power_spectrum`]
            with default arguments.
            Keys:
                map_names : sequence
                    Sequence of map names from {'local_similarity', 'local_correlation', 'power_spectrum'}.
                `map_name` : dict, optional
                    Parameters for the corresponding map.

        Returns
        -------
        dict
            Dictionary with `map_names` as keys and computed maps as values.
        """
        maps = {'local_similarity': cls.local_similarity_map,
                'local_correlation': cls.local_correlation_map,
                'power_spectrum': cls.fourier_power_spectrum}

        map_names = maps.keys() if config is None else config['map_names']
        if not np.all([map_name in maps for map_name in map_names]):
            raise ValueError

        config = config if config is not None else {}
        returns = {}
        for map_name in map_names:
            config[map_name] = {} if map_name not in config else config[map_name]
            returns[map_name] = maps[map_name](image, prediction, map_to=map_to, **config[map_name])
        return returns

    @classmethod
    def local_correlation_map(cls, image, prediction, map_to, window_size=9, n_dims=1):
        """Local correlation map between an image and estimated noise.

        Parameters
        ----------
        image : np.ndarray
        prediction : np.ndarray
        window_size : int
            if `n_dims` is 1, correlation is measured between corresponding parts of traces of `window_size` size.
            if `n_dims` is 2, correlation is measured between flattened windows of size (`window_size`, `window_size`).
        n_dims : int
            Number of dimensions for `window_size`.

        Returns
        -------
        np.ndarray
            Array of the same shape.
        """
        image = image.squeeze()
        prediction = prediction.squeeze()
        image_noise = np.abs(image - prediction)
        image = image if map_to == 'image' else prediction
        img_shape = image.shape

        # "same" padding along trace for 1d window or both dims for 2d
        pad = window_size // 2
        pad_width = [[pad, window_size - (1 + pad)], [pad * (n_dims - 1), (window_size - (1 + pad)) * (n_dims - 1)]]

        image = np.pad(image, pad_width=pad_width, mode='mean')
        image_noise = np.pad(image_noise, pad_width=pad_width, mode='mean')

        # vectorization
        window_shape=[window_size, window_size if n_dims == 2 else 1]
        image_view = np.lib.stride_tricks.sliding_window_view(image, window_shape=window_shape)
        image_noise_view = np.lib.stride_tricks.sliding_window_view(image_noise, window_shape=window_shape)

        straighten = (np.dot(*image_view.shape[:2]), np.dot(*image_view.shape[2:]))
        image_view = image_view.reshape(straighten)
        image_noise_view = image_noise_view.reshape(straighten)

        pearson = cls._pearson_corr_2d(image_view, image_noise_view).reshape(img_shape)
        return np.nan_to_num(pearson)

    @classmethod
    def _pearson_corr_2d(cls, x, y):
        """Squared Pearson correlation coeffitient between corresponding rows of 2d input arrays"""
        x_centered = x - x.mean(axis=1).reshape(-1, 1)
        y_centered = y - y.mean(axis=1).reshape(-1, 1)
        corr = (x_centered * y_centered).sum(axis=1)
        corr /= np.sqrt((x_centered**2).sum(axis=1) * (y_centered**2).sum(axis=1))
        return corr ** 2

    @classmethod
    def local_similarity_map(cls, image, prediction, map_to, lamb=0.5, window_size=9, n_dims=1, **kwargs):
        """Local Similarity Map between an image and estimated noise.
        Chen, Yangkang, and Sergey Fomel. "`Random noise attenuation using local signal-and-noise orthogonalization
        <https://library.seg.org/doi/10.1190/geo2014-0227.1>`_"

        Parameters
        ----------
        image : np.ndarray
        prediction : np.ndarray
        lamb : float
            Regularization parameter from 0 to 1.
        window_size : int
            if `n_dims` is 1, similarity is measured between corresponding parts of traces of `window_size` size.
            if `n_dims` is 2, similarity is measured between flattened windows of size (`window_size`, `window_size`).
        n_dims : int
            Number of dimensions for `window_size`.
        tol : float, optional
            Tolerance for `shaping_conjugate_gradient`.
        N : int, optional
            Maximum number of iterations for `shaping_conjugate_gradient`.

        Returns
        -------
        np.ndarray
            Array of the same shape.
        """
        image = image.squeeze()
        prediction = prediction.squeeze()
        image_noise = np.abs(image - prediction)
        image = image if map_to == 'image' else prediction
        img_shape = image.shape

        pad = window_size // 2
        pad_width = [[pad, window_size - (1 + pad)], [pad * (n_dims - 1), (window_size - (1 + pad)) * (n_dims - 1)]]

        image = np.pad(image, pad_width=pad_width, mode='mean')
        image_noise = np.pad(image_noise, pad_width=pad_width, mode='mean')

        window_shape=[window_size, window_size if n_dims == 2 else 1]
        image_view = np.lib.stride_tricks.sliding_window_view(image, window_shape=window_shape)
        image_noise_view = np.lib.stride_tricks.sliding_window_view(image_noise, window_shape=window_shape)

        straighten = (np.dot(*image_view.shape[:2]), np.dot(*image_view.shape[2:]))
        image_view = image_view.reshape(straighten)
        image_noise_view = image_noise_view.reshape(straighten)

        H = np.eye(window_size**n_dims, dtype=np.float) * lamb
        H = np.lib.stride_tricks.as_strided(H, shape=(image_view.shape[0], window_size**n_dims, window_size**n_dims),
                                                strides=(0, 8 * window_size**n_dims, 8))

        sim_local = cls._local_similarity(image_view, image_noise_view, H, **kwargs)
        return sim_local.reshape(img_shape)

    @classmethod
    def _shaping_conjugate_gradient(cls, L, H, d, tol=1e-5, N=20):
        """Vectorised Shaping Conjugate gradient Algorithm for a system with smoothing operator.
        Fomel, Sergey. "`Shaping regularization in geophysical-estimation problems
        <https://library.seg.org/doi/10.1190/1.2433716>`_"
        """
        p = np.zeros_like(d)
        m = np.zeros_like(d)
        r = -d
        sp = np.zeros_like(d)
        sm = np.zeros_like(d)
        sr = np.zeros_like(d)
        EPS = 1e-5
        for i in range(N):
            gm = (np.transpose(L, axes=[0, 2, 1]) @ r[..., np.newaxis]).squeeze() - m
            gp = (np.transpose(H, axes=[0, 2, 1]) @ gm[..., np.newaxis]).squeeze() + p
            gm = H @ gp[..., np.newaxis]
            gr = L @ gm

            rho = np.sum(gp ** 2, axis=1)
            if i == 0:
                beta = np.zeros((L.shape[0], 1))
                rho0 = rho
            else:
                beta = (rho / (rho_hat + EPS))[..., np.newaxis]
                if np.all(beta < tol) or np.all(rho / (rho0 + EPS) < tol):
                    return m

            sp = gp + beta * sp
            sm = gm.squeeze() + beta * sm
            sr = gr.squeeze() + beta * sr

            alpha = rho / (np.sum(sr ** 2, axis=1) + np.sum(sp ** 2, axis=1) - np.sum(sm ** 2, axis=1) + EPS)
            alpha = alpha[..., np.newaxis]

            p -= alpha * sp
            m -= alpha * sm
            r -= alpha * sr
            rho_hat = rho
        return m

    @classmethod
    def _local_similarity(cls, a, b, H, *args, **kwargs):
        """Local Similarity between an image and estimated noise."""
        A = np.array([np.diag(a[i]) for i in range(len(a))])
        B = np.array([np.diag(b[i]) for i in range(len(b))])
        c1 = cls._shaping_conjugate_gradient(A, H, b, *args, **kwargs)
        c2 = cls._shaping_conjugate_gradient(B, H, a, *args, **kwargs)
        return np.sum(c1 * c2, axis=1)

    @classmethod
    def fourier_power_spectrum(cls, image, prediction, fourier_map='pred', map_to=None, **kwargs):
        """Fourier Power Spectrum for an image.

        Parameters
        ----------
        fourier_map: str
            If 'image', computes power spectrum for a source image.
            If 'pred', computes power spectrum for a predicted image.

        Returns
        -------
        np.ndarray
            Array of the same shape.
        """
        image = image if fourier_map == 'image' else prediction
        image = image.squeeze()
        img_fft = fftpack.fft2(image, **kwargs)
        shift_fft = fftpack.fftshift(img_fft)
        spectrum = np.abs(shift_fft)**2
        return np.log10(spectrum).squeeze()
