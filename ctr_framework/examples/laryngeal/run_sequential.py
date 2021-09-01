import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None
from ctr_framework.ctrseq_group import CtrseqGroup
from lsdo_viz.api import Problem
from ctr_framework.mesh import trianglemesh
from ctr_framework.initpt import initialize_pt
from ctr_framework.collision_check import collision_check
from ctr_framework.log import log
import shutil
import time
from ctr_framework.equofplane import equofplane
from ctr_framework.findcircle import findCircle
from ctr_framework.design_method.seq_opt import seq_opt

#########################################
############## initialization ###########
#########################################

for j in range(1):
    # number of waypoints
    viapts_nbr= 6
    orient_nbr = 9
    k = 1
    # number of links                              
    num_nodes = 100
    # number of tubes
    tube_nbr = 4

    # initial robot configuration
    # Tube 1(inner tube) ID, OD
    '''d1 = 0.65
    d2 = 0.88
    # Tube 2 
    d3 = 1.076
    d4 = 1.296
    # Tube 3(outer tube)
    d5 = 1.470
    d6 = 2.180
    # tube guide
    d7 = 2.2 
    d8 = 2.7
    # Tube curvature (kappa)
    kappa_init = np.array([0.031, 0.0021,0.0031,0.012]).reshape((1,tube_nbr))
    # The length of tubes
    tube_length_init = np.array([305, 250,150,65]).reshape((1,tube_nbr))
    # The length of straight section of tubes
    tube_straight_init = np.array([295, 230,100,15]).reshape((1,tube_nbr))
    # joint variables
    alpha_init = np.zeros((k,tube_nbr))
    alpha_init[:,0] = -np.pi/2 -1.57
    alpha_init[:,1] = np.pi - 1.57
    alpha_init[:,2] = -np.pi/2 -1.57
    alpha_init[:,3] = -np.pi/2 -1.57
    #alpha_init = beta_init + 2 * np.random.random((k,tube_nbr)) * (-1)
    beta_init = np.zeros((k,tube_nbr))
    # beta_init[:,0] = -85
    # beta_init[:,1] = -90
    # beta_init[:,2] = -65
    # beta_init[:,3] = -30
    beta_init[:,0] = -89
    beta_init[:,1] = -90
    beta_init[:,2] = -65
    beta_init[:,3] = -30

    # beta_init = beta_init + 2 * np.random.random((k,tube_nbr)) * (-1)
    # initial torsion 
    init_dpsi = np.random.random((k,tube_nbr)) *0.01
    rotx_ = 1e-10 
    roty_ = 1e-10
    rotz_ = 1e-10
    loc = np.ones((3,1)) * 1e-5

    mdict = {'alpha':alpha_init, 'beta':beta_init,'kappa':kappa_init,
            'tube_section_straight':tube_straight_init,'tube_section_length':tube_length_init,
            'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4, 'd5':d5, 'd6':d6, 'd7':d7, 'd8':d8,
            'rotx':rotx_,'roty':roty_ ,'rotz':rotz_ , 'loc':loc, 'initial_condition_dpsi':init_dpsi,
            }
    scipy.io.savemat('initial.mat',mdict)'''
    
    # Base frame
    # base = np.array([-10,35,20]).reshape((3,1))
    # rot = np.array([3.14,0,0]).reshape((3,1)) 
    # p_plane = np.array([[-10,35,20],[-12,20,20],\
    #                     [-20,15,20]])
    # left arm
    # base = np.array([6.5,15,0]).reshape((3,1))
    # rot = np.array([0,0,0]).reshape((3,1))
    # right arm
    # base = np.array([28.5,15,0]).reshape((3,1))
    # rot = np.array([0,0,0]).reshape((3,1))  
    base = np.array([-3,15,-35]).reshape((3,1))
    rot = np.array([0.39,0,0]).reshape((3,1))  
    
    # mesh .PLY file
    meshfile = 'right.ply'
    pathfile = 'workspace.mat'
    des_vector = np.zeros((viapts_nbr,3))
    # vec = np.array([[0,-1,1],[0,-0.5,1],[0,0,1],[0,0.5,1],[0,1,1]])
    # vec = np.array([[0,-1,1],[0,-1,1],[0,-1,1]])
    vec = np.zeros((viapts_nbr,3)) 
    vec[:,:] = np.array([0,0,1])
    '''vec[1,:] = np.array([-1,1,1])
    vec[2,:] = np.array([0,1,1])
    vec[3,:] = np.array([1,1,1])
    vec[4,:] = np.array([1,0,1])'''

    des_vector[:,:] = vec / np.linalg.norm(vec)

    seq_opt(num_nodes,viapts_nbr,orient_nbr,base,rot,meshfile,pathfile,j)


