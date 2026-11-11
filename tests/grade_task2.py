import numpy as np
from tasks import *
from numpy.testing import assert_array_equal


def test_convolve_1d():
    input_array = np.array([1, 2, 3, 4, 6])
    kernel_array = np.array([1, 0, -1])
    expected_output = np.array([-2, -2, -3])
    assert_array_equal(convolve_1d(input_array, kernel_array), expected_output)
