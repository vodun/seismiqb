""" IRAP ASCII files handler. """
#pylint: disable=redefined-builtin
import numpy as np
import pandas as pd



class IRAPHandler:
    """ IRAP ASCII files handler for reading and dumping matrices.

    This data format is used for storing grid-likable matrices in the storage.

    IRAP data file starts with the header and values.
    Values are written sequentially and grouped into 6 columns.
    Header defines grid structure in the following format:

    ```
        UNKNOWN    NY               DX      DY
        XMIN       XMAX             YMIN    YMAX
        NX         ROTATION(CCW)    ROTX    ROTY
        0   0   0   0   0   0   0
    ```

    All *X and *Y values are in the CDP coordinates system.

    Note, Petrel doesn't use UNKNOWN, ROTATION(CCW), ROTX and ROTY elements,
    so we don't evaluate their exact values for dumping files.
    """
    # File format constants
    FILL_VALUE = 9999900.0
    IRAP_COLS = 6
    FLOAT_ROUND = 6

    @classmethod
    def read(cls, path, geometry):
        """ Read matrix from the path.

        Parameters
        ----------
        path : str
            Path to IRAP ASCII data file.
        geometry : instance of :class:`~.Geometry`
            Geometry that the file corresponds to.
        """
        # Extract data from file
        header = pd.read_csv(path, nrows=3, names=range(4), delimiter=' ').values
        values = pd.read_csv(path, skiprows=4, delimiter=' ', na_values=cls.FILL_VALUE,
                             names=range(cls.IRAP_COLS)).values.flatten()

        # Fit values into matrix
        ncols, nrows = int(header[0][1]), int(header[2][0])
        values = values[:ncols*nrows].reshape(ncols, nrows)

        if (nrows == geometry.shape[0]) and (ncols == geometry.shape[1]):
            matrix = values # Grid fits geometry
        else:
            # Parse data grid
            start = cls._extract_offset(header=header, geometry=geometry)
            step = cls._extract_step(geometry=geometry, header=header, format='ordinal')

            # Fill grid values
            matrix = np.full(geometry.shape[:2], fill_value=np.nan, dtype=np.float32)
            matrix[start[0]:start[0]+ncols*step[0]:step[0], start[1]:start[1]+nrows*step[1]:step[1]] = values

        matrix[np.isclose(matrix, cls.FILL_VALUE)] = np.nan
        matrix[geometry.dead_traces_matrix] = np.nan
        return matrix

    @classmethod
    def dump_matrix(cls, matrix, path, geometry, grid_step=None, offset=None, header=None):
        """ Write grid in the IRAP file format.

        Parameters
        ----------
        matrix : np.ndarray with (N, M) shape
            Two dimensional data array to dump.
        path : str
            Path to save the grid.
        geometry : instance of :class:`~.Geometry`
            Geometry that the matrix corresponds to.
        grid_step : sequence of two ints or None
            (ilines, xlines) steps for values extraction from the matrix.
            Note, if matrix is a sparse grid and is dumped with `grid_step = (1, 1)`, then some software programs
            will visualize all empty values and grid will be invisible.
            If None, then `grid_step` will be evaluated as minimal distance between not-nan elements.
        offset : sequence of two ints or None
            Grid line start indices for ilines and xlines.
            If None, then `offset` will be evaluated as minimal line indices with not-nan elements.
        header : str or None
            Dumping IRAP file header.
            If None, header is extracted from the geometry info.
            Otherwise, use provided header.
        """
        data = np.round(matrix, cls.FLOAT_ROUND)

        # Extract data from grid
        if header is None:
            offset, grid_step = cls.eval_grid_parameters(matrix)

            data = data[offset[0]::grid_step[0], offset[1]::grid_step[1]]
            header = cls.create_header(matrix=data, geometry=geometry, grid_step=grid_step, offset=offset)

        else:
            offset = cls._extract_offset(header=header, geometry=geometry)
            grid_step = cls._extract_step(geometry=geometry, header=header, format='ordinal')

            data = data[offset[0]::grid_step[0], offset[1]::grid_step[1]]

        irap_data = np.nan_to_num(data, nan=cls.FILL_VALUE).ravel()

        # Data file has X rows and 6 cols - data shape can be non-suitable for this reshape
        n_elements_to_add = cls.IRAP_COLS - len(irap_data) % cls.IRAP_COLS

        if n_elements_to_add < 6:
            irap_data = np.append(irap_data, [np.nan]*n_elements_to_add)

        irap_data = irap_data.reshape(-1, cls.IRAP_COLS)

        with open(path, mode='w', encoding='utf-8') as dst:
            np.savetxt(dst, irap_data, fmt=f'%.{cls.FLOAT_ROUND}f',
                       delimiter=' ', newline='\n', header=header, comments='')

    # Grid structure evaluations
    @staticmethod
    def eval_grid_parameters(matrix):
        """ Evaluate grid step and line start indices (offsets) from the matrix.

        Offset is evaluated as minimal line indices with not-nan elements.
        Grid step is evaluated as minimal distance between not-nan elements.
        """
        nonzero_lines = tuple(np.argwhere(np.count_nonzero(~np.isnan(matrix), axis=axis)) for axis in (1, 0))

        offset = tuple(nonzero_lines[axis].min() for axis in range(2))
        grid_step = tuple(np.diff(nonzero_lines[axis], axis=0).min() for axis in range(2))
        return offset, grid_step

    # Header based methods
    @classmethod
    def create_header(cls, matrix, geometry, grid_step=None, offset=None):
        """ Create IRAP file header for matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Two dimensional data array for which we need to create IRAP file header.
        geometry : instance of :class:`~.Geometry`
            Geometry that the matrix corresponds to.
        grid_step : sequence of two ints or None
            (ilines, xlines) steps in ordinal coordinate system for values extraction from matrix.
            Note, if matrix is a sparse grid and is dumped with `grid_step = (1, 1)`, then some software programs
            will visualize all empty values and grid will be invisible.
            If None, then `grid_step` will be evaluated as minimal distance between not-nan elements.
        offset : sequence of two ints or None
            Grid lines start indices for ilines and xlines.
            If None, then `offset` will be evaluated as minimal line indices with not-nan elements.
        """
        if grid_step is None or offset is None:
            offset, grid_step = cls.eval_grid_parameters(matrix)

        dx, dy = cls._extract_step(geometry=geometry, format='cdp')

        offset_cdp = (offset[1] * dx, offset[0] * dy)
        grid_step_cdp = (grid_step[1] * dx, grid_step[0] * dy)

        # Write header string
        header = (
            # UNKNOWN NY  DX DY
            f"-996 {matrix.shape[0]} {grid_step_cdp[0]:.6f} {grid_step_cdp[1]:.6f}\n"
            # XMIN    XMAX   YMIN  YMAX
            f"{geometry.headers['CDP_X'].min()+offset_cdp[0]:.6f} {geometry.headers['CDP_X'].max()+offset_cdp[0]:.6f} "
            f"{geometry.headers['CDP_Y'].min()+offset_cdp[1]:.6f} {geometry.headers['CDP_Y'].max()+offset_cdp[1]:.6f}\n"
            # NX   ROTATION(CCW)    ROTX   ROTY
            f"{matrix.shape[1]} {0:.6f} {geometry.headers['CDP_X'].min():.6f} {geometry.headers['CDP_Y'].min():.6f}\n"
            "0 0 0 0 0 0 0"
        )
        return header

    # Header parse
    @staticmethod
    def _extract_offset(header, geometry):
        """ Extract offset in ordinal coordinate system. """
        offset = np.array((header[1][0], header[1][2]), dtype=np.float32).reshape(1, 2)

        offset = geometry.cdp_to_lines(offset)
        offset = geometry.lines_to_ordinals(offset)[0, :]

        offset = np.ceil(offset).astype(np.int32)
        return offset

    @staticmethod
    def _extract_step(geometry, format='cdp', header=None):
        """ Extract step in cdp/ordinal coordinate system.

        Parameters
        ----------
        geometry : instance of :class:`~.Geometry`
            Geometry for which to evaluate step.
        format : {'cdp', 'ordinal'}
            Coordinate system in which to evaluate step.
        header : str or None
            Header from which to take information about cdp coordinates step.
            Used only for conversion from 'cdp' to 'ordinal' format.
        """
        # Eval step in cdp coordinates format
        rotation_matrix = geometry.rotation_matrix

        step = (rotation_matrix[:, :2] ** 2).sum(axis=1) ** 0.5

        # Convert if needed
        if 'ordinal' in format:
            step = (round(float(header[0][3])/step[0]), round(float(header[0][2])/step[1]))
        return step
