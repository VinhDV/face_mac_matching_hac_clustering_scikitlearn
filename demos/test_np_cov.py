import numpy as np
a = [3,9,8,2]
b = [4,7,2,5]
c = [3,4,6,7]
x = np.vstack([a,b,c])
cov = np.cov(x)
print