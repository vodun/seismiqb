""" This file contains entry points for running tests. """
from .tests_run import run_test_notebook


def test_geometry(capsys, tmpdir):
    """ Run SeismicGeometry test notebook.

    This test runs ./notebooks/geometry_test.ipynb test file and show execution message and
    the most important timings for SeismicGeometry tests.

    Under the hood, this notebook create a fake seismic cube, saves it in different data formats
    and for each format run SeismicGeometry tests.

    You can manage the test notebook execution kwargs which relates to cube with parameters:

    CUBE_SHAPE : sequence of three integers
        Shape of a synthetic cube.
    SEED: int or None
        Seed used for creation of random generator (check out `np.random.default_rng`).
    """
    test_kwargs={
        # Workspace parameters
        'SAVE_LOGS_REG_EXP': ['geometry_timings_*.json'],
        'TEST_OUTPUTS': ['message', 'timings'],

        # Data creation parameters
        'CUBE_SHAPE': (1000, 200, 400),
        'SEED': 42
    }

    run_test_notebook(test_name='geometry', test_kwargs=test_kwargs, capsys=capsys, tmpdir=tmpdir)

def test_charisma(capsys, tmpdir):
    """ Run CharismaMixin tests notebook.

    This test runs ./notebooks/charisma_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube (Field), saves it and checks
    matrices savings and loadings in CHARISMA data format.

    You can manage the test notebook execution kwargs which relates to cube with parameters:

    CUBE_SHAPE : sequence of three integers
        Shape of a synthetic cube.
    SEED: int or None
        Seed used for creation of random generator (check out `np.random.default_rng`).
    """
    test_kwargs={
         # Data creation parameters
        'CUBE_SHAPE': (100, 100, 100),
        'SEED': 10
    }

    run_test_notebook(test_name='charisma', test_kwargs=test_kwargs, capsys=capsys, tmpdir=tmpdir)

def test_horizon(capsys, tmpdir):
    """ Run Horizon test notebook.

    This test runs ./notebooks/horizon_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube with horizon, saves them
    and runs Horizon tests notebooks (base, extraction, manipulations, attributes).

    You can manage the test notebook execution kwargs which relates to cube and horizon with parameters:

    SYNTHETIC_MODE : bool
        Whether to create a synthetic data (cube and horizon) or use existed, provided by paths.
    CUBE_PATH : str or None
        Path to an existed seismic cube.
        Notice that it is only used with SYNTHETIC_MODE = False.
    HORIZON_PATH : str or None
        Path to an existed seismic horizon.
        Notice that it is only used with SYNTHETIC_MODE = False.
    CUBE_SHAPE : sequence of three integers
        Shape of a synthetic cube.
    GRID_SHAPE: sequence of two integers
        Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
    SEED: int or None
        Seed used for creation of random generator (check out `np.random.default_rng`).
    """
    test_kwargs={
        # Workspace parameters
        'TEST_OUTPUTS': 'message',

        # Synthetic data creation parameters
        'SYNTHETIC_MODE': True,
        'CUBE_PATH': None,
        'HORIZON_PATH': None,
        'CUBE_SHAPE': (500, 500, 200),
        'GRID_SHAPE': (10, 10),
        'SEED': 42
    }

    run_test_notebook(test_name='horizon', test_kwargs=test_kwargs, capsys=capsys, tmpdir=tmpdir)
