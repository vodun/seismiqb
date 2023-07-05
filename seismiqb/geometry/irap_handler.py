""" IRAP ASCII files handler. """
import numpy as np



class IRAPHandler:
    """ IRAP ASCII files handler for reading and dumping matrices.

    This data format is used for storing grid-likable matrices in the storage.
    """
    # File format constants
    fill_value = 9999900.0
    header_elements = 19

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
        with open(path, mode='r', encoding='utf-8') as irap_file:
            data = irap_file.read()

        values = data.split() # We can't use `np.loadtxt` because some lines can be less than 6 elements

        # Parse header
        header = values[:cls.header_elements]

        # - Find ncols, nrows
        ncols, nrows = int(header[1]), int(header[8])

        values = np.array(values[cls.header_elements:cls.header_elements + ncols * nrows])
        values = np.array(values, dtype=np.float32).reshape(ncols, nrows)

        values[values == cls.fill_value] = np.nan

        if (nrows == geometry.shape[0]) and (ncols == geometry.shape[1]):
            matrix = values # Grid fits geometry
        else:
            # Parse data grid
            start = cls._extract_offset(header=header, geometry=geometry)
            step = cls._extract_step(header=header, geometry=geometry)

            # Create matrix
            matrix = np.empty(geometry.shape[:2], dtype=np.float32)
            matrix[:, :] = np.nan

            matrix[start[0]:start[0]+ncols*step[0]:step[0], start[1]:start[1]+nrows*step[1]:step[1]] = values

        matrix[np.isclose(matrix, cls.fill_value)] = np.nan
        matrix[geometry.dead_traces_matrix] = np.nan
        return matrix

    @classmethod
    def dump_matrix(cls, matrix, path, geometry, grid_step=None, offset=None, header=None):
        """ Write grid in the IRAP ASCII file format.

        Parameters
        ----------
        matrix : np.ndarray with (N, M) shape
            Two dimensional data array to dump.
        path : str
            Path to save the grid.
        geometry : instance of :class:`~.Geometry`
            Geometry that the matrix corresponds to.
        grid_step : sequence of two ints or None
            (ilines, xlines) steps for values extraction from matrix.
            Note, if matrix is a sparse grid and is dumped with `grid_step = (1, 1)`, then some software programs
            will visualize all empty values and grid will be invisible.
            If None, then `grid_step` will be evaluated automatically as minimal distance between not-nan elements.
        offset : sequence of two ints or None
            Grid lines start indices for ilines and xlines.
            If None, then `offset` will be evaluated automatically as minimal line indices with not-nan elements.
        header : str or None
            Dumping file header in IRAP ASCII format.
            If None, header is extracted from the geometry info.
            Otherwise, use provided header.
        """
        # Extract data from grid
        if grid_step is None or offset is None:
            offset, grid_step = cls.eval_grid_parameters(matrix)

        data = np.round(matrix, 6)

        if header is None:
            data = data[offset[0]::grid_step[0], offset[1]::grid_step[1]]
            header = cls.create_header(matrix=data, geometry=geometry, grid_step=grid_step, offset=offset)
        else:
            start = cls._extract_offset(header=header, geometry=geometry)
            grid_step = cls._extract_step(header=header, geometry=geometry)

            data = data[start[0]::grid_step[0], start[1]::grid_step[1]]

        # Extract data from grid
        irap_data = np.nan_to_num(data, nan=cls.fill_value).ravel()

        # Data file has X rows and 6 cols - data shape can be non-suitable for this reshape
        n_valid_rows = len(irap_data) // 6
        n_elements_to_add = (n_valid_rows + 1) * 6 - len(irap_data)

        if n_elements_to_add > 0:
            irap_data = np.append(irap_data, [np.nan]*n_elements_to_add)

        irap_data = irap_data.reshape(-1, 6)

        with open(path, mode='w', encoding='utf-8') as dst:
            np.savetxt(dst, irap_data, fmt='%.6f', delimiter=' ', newline='\n', header=header, comments='')

    # Grid structure evaluations
    @staticmethod
    def eval_grid_parameters(matrix):
        """ Evaluate grid step and grid lines start indices from the matrix.

        Grid step is evaluated as minimal distance between not-nan elements.
        Offset is evaluated as minimal line indices with not-nan elements.
        """
        nonzero_coords = tuple(np.argwhere(np.count_nonzero(~np.isnan(matrix), axis=axis)) for axis in range(2))
        grid_step = tuple(np.diff(nonzero_coords[axis], axis=0).min() for axis in (1, 0))
        offset = tuple(nonzero_coords[axis].min() for axis in (1, 0))
        return offset, grid_step

    # Header based methods
    @staticmethod
    def create_header(matrix, geometry, grid_step=None, offset=None):
        """ Create IRAP file header for matrix.

        Header is the string in the following format:

        ```
            UNKNOWN    NY               DX      DY
            XMIN       XMAX             YMIN    YMAX
            NX         ROTATION(CCW)    ROTX    ROTY
            0   0   0   0   0   0   0
        ```

        Note, Petrel doesn't use UNKNOWN, ROTATION(CCW), ROTX and ROTY elements,
        so we don't evaluate their exact values.

        Parameters
        ----------
        matrix : np.ndarray
            Two dimensional data array for which we need to create IRAP file header.
        geometry : instance of :class:`~.Geometry`
            Geometry that the matrix corresponds to.
        grid_step : sequence of two ints or None
            (ilines, xlines) steps for values extraction from matrix.
            Note, if matrix is a sparse grid and is dumped with `grid_step = (1, 1)`, then some software programs
            will visualize all empty values and grid will be invisible.
            If None, then `grid_step` will be evaluated automatically as minimal distance between not-nan elements.
        offset : sequence of two ints or None
            Grid lines start indices for ilines and xlines.
            If None, then `offset` will be evaluated automatically as minimal line indices with not-nan elements.
        """
        if grid_step is None or offset is None:
            offset, grid_step = cls.eval_grid_parameters(matrix)
        dx, dy = _extract_coords_delta(geometry=geometry)

        grid_step_ms = (grid_step[1] * dx, grid_step[0] * dy)
        offset_ms = (offset[1] * dx, offset[0] * dy)

        # Write header string
        header = (
            # UNKNOWN NY  DX DY
            f"-996 {matrix.shape[0]} {grid_step_ms[0]:.6f} {grid_step_ms[1]:.6f}\n"
            # XMIN    XMAX   YMIN  YMAX
            f"{geometry.headers['CDP_X'].min()+offset_ms[0]:.6f} {geometry.headers['CDP_X'].max()+offset_ms[0]:.6f} "
            f"{geometry.headers['CDP_Y'].min()+offset_ms[1]:.6f} {geometry.headers['CDP_Y'].max()+offset_ms[1]:.6f}\n"
            # NX   ROTATION(CCW)    ROTX   ROTY
            f"{matrix.shape[1]} {0:.6f} {geometry.headers['CDP_X'].min():.6f} {geometry.headers['CDP_Y'].min():.6f}\n"
            "0 0 0 0 0 0 0"
        )
        return header

    # Header parse
    @staticmethod
    def _extract_offset(header, geometry):
        """ Extract offset in ordinal coordinate system. """
        offset = np.array((header[4], header[6]), dtype=np.float32).reshape(1, 2)

        offset = geometry.cdp_to_lines(offset)
        offset = geometry.lines_to_ordinals(offset)[0, :]

        offset = np.ceil(offset).astype(np.int32)
        return offset

    @staticmethod
    def _extract_step(header, geometry):
        """ Extract step in ordinal coordinate system. """
        dx, dy = _extract_coords_delta(geometry=geometry)
        step = (round(float(header[3])/dx), round(float(header[2])/dy))
        return step

# Helpers
def _extract_coords_delta(geometry):
    """ Extract coordinates delta in CDP coordinate system. """
    rotation_matrix = geometry.rotation_matrix.round(1).astype(np.int32)

    if rotation_matrix[0, 0] != 0:
        dx, dy = rotation_matrix[0, 0], rotation_matrix[1, 1] # Need to be properly checked
    else:
        dx, dy = rotation_matrix[1, 0], rotation_matrix[0, 1]
    return dx, dy
