import numpy as np
import scipy
import os


from ctr_framework.design_method.sim_opt import sim_opt



# Initialize the number of number of links and waypoints
num_nodes = 50*2
k = 3
tube_nbr = 4
# robot initial pose 
# base = np.array([-10,35,20]).reshape((3,1))
# rot = np.array([3.14,0,0]).reshape((3,1))

base = np.array([-3,15,-35]).reshape((3,1))
rot = np.array([0.39,0,0]).reshape((3,1))
des_vector = np.zeros((3,3))
vec = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
des_vector[:,:] = vec / np.linalg.norm(vec)
# mesh .PLY file
meshfile = 'right.ply'
pathfile = 'path_right.mat'
# run simultaneous optimization
sim_opt(tube_nbr,num_nodes,k,base,rot,meshfile,pathfile,des_vector)

