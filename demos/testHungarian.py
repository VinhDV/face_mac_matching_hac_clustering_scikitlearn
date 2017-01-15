from munkres import Munkres, print_matrix
import numpy as np

def pad_to_square(a, pad_value=0):
  m = a.reshape((a.shape[0], -1))
  padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
  padded[0:m.shape[0], 0:m.shape[1]] = m
  return padded

matrix = [[5, 9, 1, 0],
                    [10, 3, 2, 0],
                    [8, 7, 4, 0],
                    [8, 6, 1, 0]]
#matrix = pad_to_square(matrix,0)
m = Munkres()
indexes = m.compute(matrix)
#print_matrix(matrix, msg='Lowest cost through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print '(%d, %d) -> %d' % (row, column, value)
print 'total cost: %d' % total