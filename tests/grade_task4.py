import numpy as np
from tasks import *
from numpy.testing import assert_array_equal

def test_convolute_2d():
    input_matrix = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 10]])
    kernel_matrix = np.array([[1, 0], [0, -1]])
    expected_output = np.array([[-3, -4], [-4, -5]])
    assert_array_equal(convolute_2d(input_matrix, kernel_matrix), expected_output)