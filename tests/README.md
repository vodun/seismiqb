# A short instruction for new tests addition

## Notebook preparation

Before you add your test notebook to the running tests list, you need to make **simple preparations**:

1. If your test saves any files, it is higly recommended to use relative paths and create a test's **own directory** for saving files.

All files will be saved in the shared `TESTS_ROOT_DIR` directory (`'seismiqb\tests\test_root_dir_*'`), so a separate directory inside the `TESTS_ROOT_DIR` will prevent from mixing up files from different tests.

2. All **changeable parameters** must be initialized on the first or on the second notebook cells. All manipulations with them must be done on cells number 3 or higher.

This is caused because the `run_notebook_test.py` inserts a new cell with parameters initialization between cells number 2 and 3.

So, the recommended notebook structure is:
* **Cell 1**: necessary imports.
* **Cell 2**: parameters initialization.
* **Other cells**: tests and additional manipulations.

## Adding a new notebook into the running tests list

When you have created and prepared your test notebook, you can add it to the running list.

For this you only need to provide a `(notebook_path, params_dict)` tuple into the `notebooks_params` variable inside the `run_notebook_test.py`.

The `params_dict` is a dictionary with optional `'inputs'` and `'outputs'` keys:
* If you need to provide **new parameters values**, you need to add them in the `'inputs'` in the dictionary format `{'parameter_name': 'parameter_value'}`.
* If you want to **print into the terminal** some variables values from the notebook (such as log messages or timings), you need to add them in the `'outputs'` in the list format `['notebook_variable_name_1', 'notebook_variable_name_2']`.

```python
notebooks_params = (
    ('path/to/the/test_notebook.ipynb', {}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'sgy'}}),
    ('path/to/the/test_notebook.ipynb', {'outputs': ['message', 'timings']}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'sgy'}, 'outputs': ['message']})
)
```

That's all, now you know how to add new tests!

# Additional information

## Good practices for test notebooks

Some recommended optional practices for creating good test notebooks are recorded in the `seismiqb/tests/template_test.ipynb`.

## More about `TESTS_ROOT_DIR`

`TESTS_ROOT_DIR` is a shared directory for saving files for **all running tests**.
If you run tests locally, then it is a directory `'seismiqb/tests/tests_root_dir_*'`.

* If all tests were executed without any failures and `REMOVE_ROOT_DIR` is True, the `TESTS_ROOT_DIR` will be removed after all tests execution.
* If there were any failures in tests and/or `REMOVE_ROOT_DIR` is False, the `TESTS_ROOT_DIR` will not be removed after all tests execution.
In this case you can check saved notebooks to find out the failure reason.

## More about `notebooks_params` variable

The important details are:
* Notebooks are executed in the order in which they are defined in the `notebooks_params`.
* If you want to execute one notebook with different parameters configurations, you need to provide all of them into the `notebooks_params` variable:

```python
notebooks_params = (
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'sgy'}}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'blosc'}}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'hdf5'}}),
)
```

```python
notebooks_params = (
    *[('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': data_format}}) for data_format in ['sgy', 'blosc', 'hdf5']],
)
```

## More about terminal output message

The `run_notebook_test.py` provides in the terminal output next information:
* Error traceback and additional error info (if there is any failure in the test notebook). The additional info is: the notebook file name and the failed cell number.
* Notebook's `'outputs'` (if any variable name is provided into the `notebooks_params` variable for the notebook).
* Test conclusion: whether the notebook with tests failed or not.

One noticeable moment, the message `Notebook execution failed` is printed in two cases:

1. There is any **failure** in the notebook. Then there must be an error traceback above this message:

```python
run_notebook_test.py ---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_38767/262262589.py in <module>
----> 1 assert False, "Test failure for the example"

AssertionError: Test failure for the example
Notebook execution failed


`example_test.ipynb` failed in the cell number 6.
```

2. The notebook **wasn't executed**. In this case there are no traceback above this message. The reason for this situation is some internal execution error such as out of memory.

```python
run_notebook_test.py ---------------------------------------------------------------------------
Notebook execution failed
```
