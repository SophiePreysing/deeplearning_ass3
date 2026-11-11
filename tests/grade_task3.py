import numpy as np
from tasks import *
from numpy.testing import assert_array_equal


def test_compute_output_size_2d():
    input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel_matrix = np.array([[1, 0], [0, -1]])
    expected_output_size = (2, 2)
    assert compute_output_size_2d(input_matrix, kernel_matrix) == expected_output_size