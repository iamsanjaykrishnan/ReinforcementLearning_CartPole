"""import numpy as np
test = np.zeros([2,4,1])
def Binning(x):
    bin_boundry = [0, 1, 2, 3, 4, 5, 6]
    out_,_ = np.histogram(x,bin_boundry)
    return out_

out = np.apply_along_axis(Binning, 2, test)
print(out)"""

import numpy as np
test = np.zeros([4,1])
test2 = np.zeros([4,1])
trial = [test,test2]
trial2 = np.asarray(trial)
print(trial2.shape)






