import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, exp
from scipy.ndimage import gaussian_filter1d

def gaussian_kernel(n):
    sigma = n/(np.sqrt(5)*2)
    r = range(-int(n/2),int(n/2)+1)
    r = [(exp(-float(x)**2/(2*sigma**2))) + 1  for x in r]
    return np.asarray(r)
#1 / (sigma * sqrt(2*pi)) * 
x = gaussian_kernel(17000)
print(x)
plt.plot(x)
plt.show()