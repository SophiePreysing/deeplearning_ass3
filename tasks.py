import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

# note: only want one convolution
# note: no padding means padding = valid (no additional 0s at boarder)

def compute_output_size_1d(input_array, kernel_array):
    if input_array.ndim != 1 or kernel_array.ndim != 1:
        raise ValueError
    else:
        return (len(input_array) - len(kernel_array) + 1)


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    l = len(input_array) - len(kernel_array) + 1
    conv = np.zeros(l)        # array of the right length with 0s, for filling later
    for i in range(l):
        a = input_array[i:len(kernel_array) + i] 
        conv[i] = a.dot(kernel_array)  # could also do a @ kernel_array 

    return conv

# note: for 1D arrays, a.dot(b) / a @ b = sum of matching positions multiplied (dot product).
# In 2D, @ / .dot() do matrix multiplication (row * column), which is NOT convolution.
# For 2D convolution: multiply matching positions and add them up → np.sum(patch * kernel)

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    if np.ndim(input_matrix) == 2 and np.ndim(kernel_matrix) == 2:
        h = input_matrix.shape[0] - kernel_matrix.shape[0] + 1 #shape gives tuple index 0 gives first value (row #)
        w = input_matrix.shape[1] - kernel_matrix.shape[1] + 1
        return (h, w)
    else:
        raise ValueError


# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    size = compute_output_size_2d(input_matrix, kernel_matrix)
    hkern = kernel_matrix.shape[0]
    wkern = kernel_matrix.shape[1]
    conv = np.zeros(size)

    for i in range(size[0]):
        for j in range(size[1]):
            a = input_matrix[i:hkern + i, j:wkern + j]
            conv[i, j] = np.sum(a * kernel_matrix)

    return conv


# -----------------------------------------------