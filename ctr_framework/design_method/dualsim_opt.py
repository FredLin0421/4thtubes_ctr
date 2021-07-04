import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None

from ctr_framework.dualctrsimul_group import DualctrsimulGroup
from lsdo_viz.api import Problem
from ctr_framework.mesh_simul import trianglemesh
from ctr_framework.initpt import initialize_pt
from ctr_framework.collision_check import collision_check
import time
from ctr_framework.equofplane import equofplane
from ctr_framework.fibonacci_sphere import fibonacci_sphere


def dualsim_opt(num_nodes,k,num_cases,base,rot,meshfile,leftpath,rightpath):
    # initialization
    via_pts = k
    pt = np.zeros((num_cases,via_pts,3))
    a = 30
    path_1 = 'leftpath.mat'
    path_2 = 'rightpath.mat'

    tar_vector = np.zeros((num_cases,2,3))
    pt[0,:,:] = initialize_pt(via_pts,path_1)
    pt[1,:,:] = initialize_pt(via_pts,path_2)

    p_plane = np.array([[-10,35,20],[-12,20,20],\
                        [-20,15,20]])
    equ_paras = equofplane(p_plane[0,:],p_plane[1,:],p_plane[2,:]) 
    # norm1_1 = np.linalg.norm(pt1[0,:]-pt1[-1,:],ord=1.125)
    # norm1_2 = np.linalg.norm(pt2[0,:]-pt2[-1,:],ord=1.125)
    k_ = 1
    alpha_ = np.zeros((num_cases,k,3))
    beta_ = np.zeros((num_cases,k,3))
    initial_condition_dpsi_ = np.zeros((num_cases,k,3))
    lag_ = np.zeros((num_cases,k,1))
    rho_ = np.zeros((num_cases,k,1))
    zeta_ = np.zeros((num_cases,k,1))
    

    count = 0
    t0 = time.time()

    # configs1 = scipy.io.loadmat('left_seq.mat')

    # lag_[0,:,:] = configs1['lag']
    # rho_[0,:,:] = configs1['rho']
    # zeta_[0,:,:] = configs1['zeta']

    # configs2 = scipy.io.loadmat('right_seq.mat')
    # lag_[1,:,:] = configs2['lag']
    # rho_[1,:,:] = configs2['rho']
    # zeta_[1,:,:] = configs2['zeta']


    for i in range(k):
        
        configl = scipy.io.loadmat('seq_l'+str(i)+'.mat')
        alpha_[0,count,:] = configl['alpha']
        beta_[0,count,:] = configl['beta']
        initial_condition_dpsi_[0,count,:] = configl['initial_condition_dpsi']
        lag_[0,count,:] = configl['lag']
        rho_[0,count,:] = configl['rho']
        zeta_[0,count,:] = configl['zeta']

        configr = scipy.io.loadmat('seq_r'+str(i)+'.mat')
        alpha_[1,count,:] = configr['alpha']
        beta_[1,count,:] = configr['beta']
        initial_condition_dpsi_[1,count,:] = configr['initial_condition_dpsi']
        lag_[1,count,:] = configr['lag']
        rho_[1,count,:] = configr['rho']
        zeta_[1,count,:] = configr['zeta']

        count = count+1
    

    mdict1 = {'alpha':alpha_[0,:,:], 'beta':beta_[0,:,:],'kappa':configl['kappa'], 'rho':rho_[0,:,:], 'lag':lag_[0,:,:], 'zeta':zeta_[0,:,:],
                        'tube_section_straight':configl['tube_section_straight'],'tube_section_length':configl['tube_section_length'],
                        'd1':configl['d1'], 'd2':configl['d2'], 'd3':configl['d3'], 'd4':configl['d4'], 'd5':configl['d5'], 'd6':configl['d6'],
                        'initial_condition_dpsi':initial_condition_dpsi_[0,:,:], 'rotx':configl['rotx'],'roty':configl['roty'],'rotz':configl['rotz'],
                        'eps_r':configl['eps_r'], 'eps_p':configl['eps_p'], 'eps_e':configl['eps_e'], 'loc':configl['loc'],
                        }
    scipy.io.savemat('siml.mat',mdict1)

    mdict2 = {'alpha':alpha_[1,:,:], 'beta':beta_[1,:,:],'kappa':configr['kappa'], 'rho':rho_[1,:,:], 'lag':lag_[1,:,:], 'zeta':zeta_[1,:,:],
                        'tube_section_straight':configr['tube_section_straight'],'tube_section_length':configr['tube_section_length'],
                        'd1':configr['d1'], 'd2':configr['d2'], 'd3':configr['d3'], 'd4':configr['d4'], 'd5':configr['d5'], 'd6':configr['d6'],
                        'initial_condition_dpsi':initial_condition_dpsi_[1,:,:], 'rotx':configr['rotx'],'roty':configr['roty'],'rotz':configr['rotz'],
                        'eps_r':configr['eps_r'], 'eps_p':configr['eps_p'], 'eps_e':configr['eps_e'], 'loc':configr['loc'], 
                        }
    scipy.io.savemat('simr.mat',mdict2)



        
    prob1 = Problem(model=DualctrsimulGroup(k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, viapts_nbr=via_pts, \
                        pt=pt, tar_vector=tar_vector,\
                        zeta = zeta_, rho=rho_, lag=lag_,\
                        rot=rot,base = base,equ_paras = equ_paras,\
                        meshfile=meshfile,leftpath=leftpath,rightpath=rightpath,\
                            ))
    prob1.driver = pyOptSparseDriver()
    # prob.driver.options['optimizer'] = 'SNOPT'
    # prob.driver = driver = pyOptSparseDriver()
    # prob.driver = om.ScipyOptimizeDriver()
    prob1.driver.options['optimizer'] = 'SNOPT'
    prob1.driver.opt_settings['Verify level'] = 0
    prob1.driver.opt_settings['Major iterations limit'] = 50 #1000
    prob1.driver.opt_settings['Minor iterations limit'] = 1000
    prob1.driver.opt_settings['Iterations limit'] = 1000000
    prob1.driver.opt_settings['Major step limit'] = 2.0
    prob1.driver.opt_settings['Major feasibility tolerance'] = 1.0e-4
    prob1.driver.opt_settings['Major optimality tolerance'] = 1.0e-3
    prob1.driver.opt_settings['Minor feasibility tolerance'] = 1.0e-4

    prob1.setup()
    prob1.run_model()
    prob1.run_driver()
    # prob1.check_partials(compact_print=True)
    # flag1,detection1 = collision_check(prob1['Ctr1Group.rot_p'],prob1['Ctr1Group.rot_d2'],prob1['Ctr1Group.rot_d4'],prob1['Ctr1Group.rot_d6'],\
    #                             prob1['Ctr1Group.rot_tube_ends'],num_nodes,mesh,k)
    # flag2,detection2 = collision_check(prob1['Ctr2Group.rot_p'],prob1['Ctr2Group.d2'],prob1['Ctr2Group.d4'],prob1['Ctr2Group.d6'],\
    #                             prob1['Ctr2Group.tube_ends'],num_nodes,mesh,k)
        
    error1 = prob1['Ctr1Group.targetnorm']
    error2 = prob1['Ctr2Group.targetnorm']


    mdict1 = {'points':prob1['Ctr1Group.integrator_group3.state:p'], 'alpha':prob1['Ctr1Group.alpha'], 'beta':prob1['Ctr1Group.beta'],'kappa':prob1['Ctr1Group.kappa'],
                    'tube_section_straight':prob1['Ctr1Group.tube_section_straight'],'tube_section_length':prob1['Ctr1Group.tube_section_length'],
                    'd1':prob1['Ctr1Group.d1'], 'd2':prob1['Ctr1Group.d2'], 'd3':prob1['Ctr1Group.d3'], 'd4':prob1['Ctr1Group.d4'], 'd5':prob1['Ctr1Group.d5'], 'd6':prob1['Ctr1Group.d6'],
                    'initial_condition_dpsi':prob1['Ctr1Group.initial_condition_dpsi'], 'loc':prob1['Ctr1Group.loc'], 'rotx':prob1['Ctr1Group.rotx'], 'roty':prob1['Ctr1Group.roty'],
                    'rotz':prob1['Ctr1Group.rotz'], 'error1':error1, 'lag':lag_,'zeta':zeta_,'rho':rho_,
                    }
    mdict2 = {'points':prob1['Ctr2Group.integrator_group3.state:p'], 'alpha':prob1['Ctr2Group.alpha'], 'beta':prob1['Ctr2Group.beta'],'kappa':prob1['Ctr2Group.kappa'],
                    'tube_section_straight':prob1['Ctr2Group.tube_section_straight'],'tube_section_length':prob1['Ctr2Group.tube_section_length'],
                    'd1':prob1['Ctr2Group.d1'], 'd2':prob1['Ctr2Group.d2'], 'd3':prob1['Ctr2Group.d3'], 'd4':prob1['Ctr2Group.d4'], 'd5':prob1['Ctr2Group.d5'], 'd6':prob1['Ctr2Group.d6'],
                    'initial_condition_dpsi':prob1['Ctr2Group.initial_condition_dpsi'], 'loc':prob1['Ctr2Group.loc'], 'rotx':prob1['Ctr2Group.rotx'], 'roty':prob1['Ctr2Group.roty'],
                    'rotz':prob1['Ctr2Group.rotz'],'error2':error2, 'lag':lag_,'zeta':zeta_,'rho':rho_,
                    }

    scipy.io.savemat('leftarm_f'+str(3)+'.mat',mdict1)

    scipy.io.savemat('rightarm_f'+str(3)+'.mat',mdict2)


    t1 = time.time()
    print(t1-t0)
