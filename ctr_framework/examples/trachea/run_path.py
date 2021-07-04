import numpy as np
import scipy
from ctr_framework.design_method.path_opt import path_opt
# from path_opt import path_opt


# Initialize the number of control points and path points
num_cp = 25
num_pt = 100
# User-defined start point and target point
# left arms
# sp = np.array([6.5, 15, 20])
# fp = np.array([12.5, 15, 150])
# right arms
sp = np.array([28.5, 15, 20]);
fp = np.array([22.5, 15, 150]);

# sp = np.array([-10,35,0])
# fp = np.array([-10,-33,-103])

# mesh .PLY file
filename = 'larynscope.ply'

path_opt(num_cp,num_pt,sp,fp,filename)



