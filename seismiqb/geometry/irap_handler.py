""" IRAP ASCII files handler. """
import numpy as np



class _IRAPHandler:
    """ IRAP ASCII files handler for reading and dumping matrices. """
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
        with open(path, mode='r') as irap_file:
            data = irap_file.read()

        values = data.split() # We can't use `np.loadtxt` because some lines can be less than 6 elements

        # Parse header
        header = values[:cls.header_elements]

        # - Find ncols, nrows
        ncols, nrows = int(header[1]), int(header[8])
        values = np.array(values[cls.header_elements:cls.header_elements + ncols * nrows])

        if (nrows == geometry.shape[0]) and (ncols == geometry.shape[1]):
            # Grid fits geometry
            matrix = np.array(values, dtype=np.float32).reshape(ncols, nrows)
        else:
            # Find start and stop in ordinal coordinate system
            ranges = np.array(header[4:8], dtype=np.float32).reshape(2, 2).T

            ranges = geometry.cdp_to_lines(ranges)
            ranges = geometry.lines_to_ordinals(ranges)

            ranges = np.ceil(ranges).astype(np.int32)

            start = (ranges[0, 0], ranges[0, 1])
            stop = (ranges[1, 0] + 1, ranges[1, 1] + 1)

            # Find step
            rotation_matrix = geometry.rotation_matrix.round(1).astype(np.int32)

            if rotation_matrix[0, 0] != 0:
                dx, dy = rotation_matrix[0, 0], rotation_matrix[1, 1] # Need to be properly checked
            else:
                dx, dy = rotation_matrix[1, 0], rotation_matrix[0, 1]

            step = (round(float(header[3])/dx), round(float(header[2])/dy))

            # Create matrix
            matrix = np.empty(geometry.shape[:2], dtype=np.float32)
            matrix[:, :] = np.nan
            mgrid = np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]]
            matrix[mgrid[0], mgrid[1]] = np.array(values, dtype=np.float32).reshape(ncols, nrows)

        matrix[np.isclose(matrix, cls.fill_value)] = np.nan
        matrix[geometry.dead_traces_matrix] = np.nan
        return matrix

    @classmethod
    def dump_matrix(cls, matrix, path, geometry, header=None):
        """ Write matrix in the IRAP ASCII file format.

        Parameters
        ----------
        matrix : np.ndarray
            Two dimensional data array.
        path : str
            Path to save the matrix.
        geometry : instance of :class:`~.Geometry`
            Geometry that the matrix corresponds to.
        header : str or None
            Dumping file header in IRAP ASCII format.
            If None, header is extracted from the geometry info.
            Otherwise, use provided header.
        """
        grid_step = (1, 1) # TODO: make with other steps
        header = header or cls.create_header(matrix=matrix, geometry=geometry, grid_step=grid_step)

        data = np.round(matrix, 6)

        irap_data = np.nan_to_num(data, nan=cls.fill_value)
        irap_data = irap_data.ravel()

        # Data file has X rows and 6 cols - data can be n
        n_valid_rows = len(irap_data) // 6

        if n_valid_rows * 6 < len(irap_data):
            n_add_elements = (n_valid_rows + 1) * 6 - len(irap_data)
            np.extend(irap_data, [np.nan]*n_add_elements)

        irap_data = irap_data.reshape(-1, 6)

        with open(path, mode='w') as dst:
            np.savetxt(dst, irap_data, fmt='%.6f', delimiter=' ', newline='\n', header=header, comments='')

    @staticmethod
    def create_header(matrix, geometry, grid_step=(1, 1)):
        """ Create IRAP file header for matrix.

        Header is string in the following format:

        ```
            UNKNOWN    NY               DX      DY
            XMIN       XMAX             YMIN    YMAX
            NX         ROTATION(CCW)    ROTX    ROTY
            0   0   0   0   0   0   0
        ```

        Note, Petrel doesn't use UNKNOWN, ROTATION(CCW), ROTX and ROTY elements, so we don't extract them correctly.

        Parameters
        ----------
        matrix : np.ndarray
            Two dimensional data array for which we need to create IRAP file header.
        geometry : instance of :class:`~.Geometry`
            Geometry that the matrix corresponds to.
        """
        # Find dx dy
        rotation_matrix = geometry.rotation_matrix.round(1).astype(np.int32)

        if rotation_matrix[0, 0] != 0:
            dx, dy = rotation_matrix[0, 0], rotation_matrix[1, 1]
        else:
            dx, dy = rotation_matrix[1, 0], rotation_matrix[0, 1]

        # Write header string
        header = (
            # UNKNOWN NY  DX DY
            f"-996 {matrix.shape[0]} {dx * grid_step[0]:.6f} {dy * grid_step[1]:.6f}\n"
            # XMIN    XMAX   YMIN  YMAX
            f"{geometry.headers['CDP_X'].min():.6f} {geometry.headers['CDP_X'].max():.6f} "
            f"{geometry.headers['CDP_Y'].min():.6f} {geometry.headers['CDP_Y'].max():.6f}\n"
            # NX   ROTATION(CCW)    ROTX   ROTY
            f"{matrix.shape[1]} {0:.6f} {geometry.headers['CDP_X'].min():.6f} {geometry.headers['CDP_Y'].min():.6f}\n"
            "0   0   0   0   0   0   0"
        )
        return header

IRAPHandler = _IRAPHandler
