import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None

# from ctrviz_group import CtrvizGroups
from ctr_framework.ctrsimul_group import CtrsimulGroup
from lsdo_viz.api import Problem
from ctr_framework.mesh_simul import trianglemesh
from ctr_framework.initpt import initialize_pt
from ctr_framework.collision_check import collision_check
import time
from ctr_framework.equofplane import equofplane
from ctr_framework.fibonacci_sphere import fibonacci_sphere
from ctr_framework.log import log



def sim_opt(num_nodes,k,base,rot,meshfile,pathfile):
    
    mesh = trianglemesh(num_nodes,k,meshfile)
    a = 30
    # robot initial pose trachea
    
    pt = initialize_pt(k,pathfile)
    pt_pri =  initialize_pt(k * 2,pathfile)
    # find 3 points on the plane
    # p_plane = np.array([[-13.2501,-22.5262,110.735],[-12.6813,-26.3715,98.0471],\
    #                     [-19.8698,-25.6478,103.586]])
    p_plane = np.array([[-10,35,20],[-12,20,20],\
                        [-20,15,20]])
    equ_paras = equofplane(p_plane[0,:],p_plane[1,:],p_plane[2,:]) 
    norm1 = np.linalg.norm(pt[0,:]-pt[-1,:],ord=1.125)


    # pt_pri =  initialize_pt(k * 2)
    # opt_tol = [1e-2,1e-3]
    'step 3: final optimization'
    k = 10
    k_ = 1
    alpha_ = np.zeros((k,3))
    beta_ = np.zeros((k,3))
    initial_condition_dpsi_ = np.zeros((k,3))
    lag_ = np.zeros((k,1))
    rho_ = np.zeros((k,1))
    zeta_ = np.zeros((k,1))

    count = 0
    for i in range(k):
        
        # configs = scipy.io.loadmat('seq_'+str(i)+'.mat')
        configs = scipy.io.loadmat('seq_l'+str(i)+'.mat')
        alpha_[count,:] = configs['alpha']
        beta_[count,:] = configs['beta']
        initial_condition_dpsi_[count,:] = configs['initial_condition_dpsi']
        lag_[count,:] = configs['lag']
        rho_[count,:] = configs['rho']
        zeta_[count,:] = configs['zeta']
        count = count+1
    mdict1 = {'alpha':alpha_, 'beta':beta_,'kappa':configs['kappa'], 'rho':rho_, 'lag':lag_, 'zeta':zeta_,
                        'tube_section_straight':configs['tube_section_straight'],'tube_section_length':configs['tube_section_length'],
                        'd1':configs['d1'], 'd2':configs['d2'], 'd3':configs['d3'], 'd4':configs['d4'], 'd5':configs['d5'], 'd6':configs['d6'],
                        'initial_condition_dpsi':initial_condition_dpsi_, 'rotx':configs['rotx'],'roty':configs['roty'],'rotz':configs['rotz'],
                        'eps_r':configs['eps_r'], 'eps_p':configs['eps_p'], 'eps_e':configs['eps_e'], 'loc':configs['loc'],
                        }
    scipy.io.savemat('simul.mat',mdict1)

    flag=1
    error=np.ones((k,1))
    i=0
    lag = np.ones((k,1))
    multiplier_zeta = 1 
    multiplier_rho = 1
    jointvalues_adrs = 'simul.mat'
    while flag==1 or error[-1]>5:

        if flag==1 and i>=0:
            multiplier_zeta = 50*i
            zeta_ = multiplier_zeta + zeta_
        
        prob1 = Problem(model=CtrsimulGroup(k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, \
                            pt=pt[:,:], meshfile=meshfile, jointvalues_adrs = jointvalues_adrs,\
                            pt_full = pt, viapts_nbr=k, zeta = zeta_, rho=rho_, lag=lag_,\
                            rotx_init=rot[0],roty_init=rot[1], rotz_init=rot[2],base = base,equ_paras = equ_paras))
        i+=1
        prob1.driver = pyOptSparseDriver()
        prob1.driver.options['optimizer'] = 'SNOPT'
        prob1.driver.opt_settings['Major iterations limit'] = 20 #1000
        prob1.driver.opt_settings['Minor iterations limit'] = 1000
        prob1.driver.opt_settings['Iterations limit'] = 1000000
        prob1.driver.opt_settings['Major step limit'] = 2.0
        prob1.driver.opt_settings['Major feasibility tolerance'] = 1.0e-4
        prob1.driver.opt_settings['Major optimality tolerance'] = 1.0e-3
        prob1.driver.opt_settings['Minor feasibility tolerance'] = 1.0e-4

        prob1.setup()
        prob1.run_model()
        prob1.run_driver()
        flag,detection = collision_check(prob1['rot_p'],prob1['d2'],prob1['d4'],prob1['d6'],\
                                    prob1['tube_ends'],num_nodes,mesh,k)
        error = prob1['targetnorm']

        if error[-1,:] >= 5:
            multiplier_rho = error
            rho_ = multiplier_rho + rho_
            lag_ = lag_ + (rho_) * error/norm1 
        mdict2 = {'points':prob1['integrator_group3.state:p'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                        'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                        'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                        'initial_condition_dpsi':prob1['initial_condition_dpsi'], 'loc':prob1['loc'], 'rotx':prob1['rotx'], 'roty':prob1['roty'],
                        'rotz':prob1['rotz'],
                        'rho':rho_,'lag':lag_,'zeta':zeta_, 'eps_r':configs['eps_r'], 'eps_p':configs['eps_p'], 'eps_e':configs['eps_e'],
                        'error':prob1['targetnorm'], 'tip_position':prob1['desptsconstraints'],
                        }

        scipy.io.savemat('siml_f_'+str(i)+'.mat',mdict2)
    

