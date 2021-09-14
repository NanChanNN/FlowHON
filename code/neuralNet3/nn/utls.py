import numpy as np

def check_unique_axis(mat, axis=0):
    sum_col = np.sum(mat, axis=axis)
    bool_col = np.abs(sum_col - 1.0) < 1e-05
    assert np.all(bool_col) == True