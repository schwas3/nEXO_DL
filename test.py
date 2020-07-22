import scipy.misc
import numpy as np
#ar1 = np.load('/home/zl423/project/dlphysics/nEXO_DL_float16/sens_npy/gamma217.root_1369.npy')
ar1 = np.load('/home/zl423/project/dlphysics/nEXO_DL_float16/sens_npy/bb0n377.root_163.npy')
for i in range(200):
    for k in range(2):
        if max(ar1[i, :, k]) > 0:
            print(ar1[i, :, k])
print(ar1.dtype)
