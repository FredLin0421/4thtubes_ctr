import numpy as np
import scipy
import os


from ctr_framework.design_method.dualsim_opt import dualsim_opt



# Initialize the number of number of links and waypoints
num_nodes = 50
k = 5
num_cases = 2
# robot initial pose 
# rotx = np.zeros((num_cases))
# roty = np.zeros((num_cases))
# rotz = np.zeros((num_cases))
rot = np.zeros((num_cases,3))
base = np.zeros((num_cases,3))
rot[0,0] = 0
rot[0,1] = 0
rot[0,2] = 0
base[0,:] = np.array([6.5,15,0]).reshape((3,))
rot[1,0] = 0
rot[1,1] = 0
rot[1,2] = 0
base[1,:] = np.array([28.5,15,0]).reshape((3,))

# mesh .PLY file
meshfile = 'larynscope2.ply'
leftpath = 'leftpath.mat'
rightpath = 'rightpath.mat' 

# run simultaneous optimization
dualsim_opt(num_nodes,k,num_cases,base,rot,meshfile,leftpath,rightpath)

