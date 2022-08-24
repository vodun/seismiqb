""" Metrics for denoising seismic data. """
import numpy as np
from scipy import fftpack
import torch

from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio, mean_squared_error,\
                                    error_relative_global_dimensionless_synthesis, universal_image_quality_index

METRICS = {'ssim': structural_similarity_index_measure,
           'psnr': peak_signal_noise_ratio,
           'ergas': error_relative_global_dimensionless_synthesis,
           'uqi': universal_image_quality_index,
           'mse': mean_squared_error}


class DenoisingMetrics:
    """ Metrics assotiated with denoising performance estimation. """
    @classmethod
    def evaluate(cls, images, predictions, objective='metrics', metrics='all'):
        """ Compute metrics related to denoising performance.

        Parameters
        ----------
        objective : str
            If 'metrics', then evaluate metrics for a batch of images.
            If 'maps', then compute maps for a single provided image or first image from the batch.
        images : np.ndarray or torch.Tensor
            Source images to evaluate. Should be of shape [B, H, W]. If `objective` is 'maps', also supports [H, W].
        predictions : np.ndarray or torch.Tensor
            Predicted images. Should be of the same shape as `images`.
        metrics : dict, list or str
            Specifies functions to compute and their parameters. Should correspond to `objective`:
            if `objective` is 'metrics', then consists of metric names from {'ssim', 'psnr', 'ergas', 'uqi', 'mse'},
            elif `objective` is 'maps', then consists of map names from
            {'local_similarity', 'local_correlation', 'power_spectrum'}.

            If dict, then should contain metric names as keys and their parameters as values.
            If list, then consists of metric names, and they are evaluated with default parametres.
            If 'all', then evaluate all metrics for corresponding `objective` with default parameters.

        Returns
        -------
        dict
            Dictionary with metric names as keys and computed metrics as values.
        """

        MAPS = {'local_similarity': cls.local_similarity_map,
                'local_correlation': cls.local_correlation_map,
                'power_spectrum': cls.fourier_power_spectrum}

        if objective == 'metrics':
            evaluate = METRICS
            images = images if isinstance(images, torch.Tensor) else torch.tensor(images)
            predictions = predictions if isinstance(predictions, torch.Tensor) else torch.tensor(predictions)
        elif objective == 'maps':
            evaluate = MAPS
            images = images[0] if len(images.squeeze().shape) == 3 else images
            predictions = predictions[0] if len(predictions.squeeze().shape) == 3 else predictions
        else:
            raise ValueError('Incorrect objective')

        if isinstance(metrics, dict):
            metric_names = metrics.keys()
        elif isinstance(metrics, list):
            metric_names = metrics
            metrics = {name: {} for name in metric_names}
        elif metrics == 'all':
            metric_names = evaluate.keys()
            metrics = {name: {} for name in metric_names}
        else:
            raise ValueError('Incorrect configuration')

        if not all(metric in evaluate for metric in metric_names):
            raise ValueError('Incorrect metric name')

        returns = {}
        for metric in metric_names:
            kwargs = metrics.get(metric, {})
            res = evaluate[metric](images, predictions, **kwargs)
            returns[metric] = res.item() if isinstance(res, torch.Tensor) else res
        return returns

    @classmethod
    def local_correlation_map(cls, image, prediction, map_to='pred', window_size=9, n_dims=1):
        """Local correlation map between an image and estimated noise.
        Parameters
        ----------
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
        """ Squared Pearson correlation coefficient between corresponding rows of 2d input arrays. """
        x_centered = x - x.mean(axis=1, keepdims=True).reshape(-1, 1)
        y_centered = y - y.mean(axis=1, keepdims=True).reshape(-1, 1)
        corr = (x_centered * y_centered).sum(axis=1)
        corr /= np.sqrt((x_centered**2).sum(axis=1) * (y_centered**2).sum(axis=1))
        return corr ** 2

    @classmethod
    def local_similarity_map(cls, image, prediction, map_to='pred', lamb=0.5, window_size=9, n_dims=1, **kwargs):
        """ Local Similarity Map between an image and estimated noise.
        Chen, Yangkang, and Sergey Fomel. "`Random noise attenuation using local signal-and-noise orthogonalization
        <https://library.seg.org/doi/10.1190/geo2014-0227.1>`_"

        Parameters
        ----------
        lamb : float
            Regularization parameter from 0 to 1.
        window_size : int
            Size of the window for a local similarity estimation.
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

        sim_local = cls._local_similarity(a=image_view, b=image_noise_view, H=H, **kwargs)
        return sim_local.reshape(img_shape)

    @classmethod
    def _shaping_conjugate_gradient(cls, L, H, d, tol=1e-5, N=20):
        """ Vectorised Shaping Conjugate gradient Algorithm for a system with smoothing operator.
        Fomel, Sergey. "`Shaping regularization in geophysical-estimation problems
        <https://library.seg.org/doi/10.1190/1.2433716>`_".
        Variables and parameters are preserved as in the paper.
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
        """ Local Similarity between an image and estimated noise. """
        A = np.array([np.diag(a[i]) for i in range(len(a))])
        B = np.array([np.diag(b[i]) for i in range(len(b))])
        c1 = cls._shaping_conjugate_gradient(L=A, H=H, d=b, *args, **kwargs)
        c2 = cls._shaping_conjugate_gradient(L=B, H=H, d=a, *args, **kwargs)
        return np.sum(c1 * c2, axis=1)

    @classmethod
    def fourier_power_spectrum(cls, image, prediction, fourier_map='pred', map_to=None, **kwargs):
        """ Fourier Power Spectrum for an image.

        Parameters
        ----------
        fourier_map : str
            If 'image', computes power spectrum for `image`.
            If 'pred', computes power spectrum for `prediction`.

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
