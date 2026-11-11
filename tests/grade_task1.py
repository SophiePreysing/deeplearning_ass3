import numpy as np
from tasks import *
from numpy.testing import assert_array_equal

def test_compute_output_size_1d():
    input_array = np.array([1, 2, 3, 4, 5])
    kernel_array = np.array([1, 0, -1])
    expected_output_length = 3
    assert compute_output_size_1d(input_array, kernel_array) == expected_output_length