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



def sim_opt(tube_nbr,num_nodes,k,base,rot,meshfile,pathfile):
    
    mesh = trianglemesh(num_nodes,k,meshfile)
    a = 30
    # robot initial pose trachea
    
    #pt = initialize_pt(k,pathfile)
    #pt_full = initialize_pt(k,pathfile)
    # find 3 points on the plane
    # p_plane = np.array([[-13.2501,-22.5262,110.735],[-12.6813,-26.3715,98.0471],\
    #                     [-19.8698,-25.6478,103.586]])
    p_plane = np.array([[-10,35,20],[-12,20,20],\
                        [-20,15,20]])
    equ_paras = equofplane(p_plane[0,:],p_plane[1,:],p_plane[2,:])
    


    # pt_pri =  initialize_pt(k * 2)
    # opt_tol = [1e-2,1e-3]
    'step 3: final optimization'
    k_ = 1
    alpha_ = np.zeros((k,tube_nbr))
    beta_ = np.zeros((k,tube_nbr))
    initial_condition_dpsi_ = np.zeros((k,tube_nbr))
    lag_ = np.zeros((k,1))
    rho_ = np.zeros((k,1))
    zeta_ = np.zeros((k,1))
    kappa = np.zeros((k,tube_nbr))
    tube_section_straight = np.zeros((k,tube_nbr))
    tube_section_length = np.zeros((k,tube_nbr))
    d1 = np.zeros((k))
    d2 = np.zeros((k))
    d3 = np.zeros((k))
    d4 = np.zeros((k))
    d5 = np.zeros((k))
    d6 = np.zeros((k))
    d7 = np.zeros((k))
    d8 = np.zeros((k))
    des_vector = np.zeros((k,3))
    count = 0
    eps_e = 1
    eps_r = 1
    eps_p = 1
    # pt = np.zeros((k,3))    
    # pt[0,:] = np.array([-3,5,182.5])
    # pt[1,:] = np.array([-3,0,185])
    # pt[2,:] = np.array([-3,-5,182.5])
    workspace = scipy.io.loadmat('workspace_f.mat')
    # pt = np.zeros((k,3))
    # pt[:,:] = np.tile(workspace['pt'],(9,1))
    pt = workspace['pt']
    norm1 = np.linalg.norm(workspace['pt'][0,:]-workspace['pt'][-1,:],ord=1.125)
    for i in range(9):
        for j in range(27):
            # configs = scipy.io.loadmat('seq_'+str(i)+'.mat')
            configs = scipy.io.loadmat('init_'+str(i+1)+'.mat')
            alpha_[count,:] = configs['alpha']
            beta_[count,:] = configs['beta']
            initial_condition_dpsi_[count,:] = configs['initial_condition_dpsi']
            # lag_[count,:] = configs['lag']
            # rho_[count,:] = configs['rho']
            # zeta_[count,:] = configs['zeta']
            lag_[count,:] = 1
            rho_[count,:] = 20
            zeta_[count,:] = 1e-2
            kappa[count,:] = configs['kappa']
            tube_section_length[count,:] = configs['tube_section_length'].squeeze()
            tube_section_straight[count,:] = configs['tube_section_straight'].squeeze()
            d1[count] = configs['d1']
            d2[count] = configs['d2']
            d3[count] = configs['d3']
            d4[count] = configs['d4']
            d5[count] = configs['d5']
            d6[count] = configs['d6']
            d7[count] = configs['d7']
            d8[count] = configs['d8']
            des_vector[count,:] = configs['des_vector']
            count = count+1
    
    # mdict1 = {'alpha':alpha_, 'beta':beta_,'kappa':np.sum(kappa,0)/k,'rho':rho_, 'lag':lag_, 'zeta':zeta_,
    #                     'tube_section_straight':np.sum(tube_section_straight,0)/k,'tube_section_length':np.sum(tube_section_length,0)/k,
    #                     'd1':np.sum(d1)/k, 'd2':np.sum(d2)/k, 'd3':np.sum(d3)/k, 'd4':np.sum(d4)/k, 'des_vector':des_vector,
    #                     'd5':np.sum(d5)/k, 'd6':np.sum(d6)/k,'d7':np.sum(d7)/k, 'd8':np.sum(d8)/k,
    #                     'initial_condition_dpsi':initial_condition_dpsi_, 'rotx':configs['rotx'],'roty':configs['roty'],'rotz':configs['rotz'],
    #                     'eps_r':configs['eps_r'], 'eps_p':configs['eps_p'], 'eps_e':configs['eps_e'], 'loc':configs['loc'],
    #                     }
    mdict1 = {'alpha':alpha_, 'beta':beta_,'kappa':np.sum(kappa,0)/k,'rho':rho_, 'lag':lag_, 'zeta':zeta_,
                        'tube_section_straight':np.sum(tube_section_straight,0)/k,'tube_section_length':np.sum(tube_section_length,0)/k,
                        'd1':np.sum(d1)/k, 'd2':np.sum(d2)/k, 'd3':np.sum(d3)/k, 'd4':np.sum(d4)/k, 'des_vector':des_vector,
                        'd5':np.sum(d5)/k, 'd6':np.sum(d6)/k,'d7':np.sum(d7)/k, 'd8':np.sum(d8)/k,
                        'initial_condition_dpsi':initial_condition_dpsi_, 'rotx':configs['rotx'],'roty':configs['roty'],'rotz':configs['rotz'],
                        'eps_r':eps_r, 'eps_p':eps_p, 'eps_e':eps_e, 'loc':configs['loc']+1e-10,
                        }
    scipy.io.savemat('simul.mat',mdict1)

    flag=1
    error=np.ones((k,1))
    i=0
    lag = np.ones((k,1))
    multiplier_zeta = 1 
    multiplier_rho = 1
    jointvalues_adrs = 'simul.mat'
    count1 = 0
    while flag==1 or error.any()>3:

        if flag==1 and i>=0:
            multiplier_zeta = 50*i
            zeta_ = multiplier_zeta + zeta_
        
        prob1 = Problem(model=CtrsimulGroup(tube_nbr=tube_nbr,k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, \
                            pt=pt[:,:], meshfile=meshfile, jointvalues_adrs = jointvalues_adrs,\
                            pt_full = pt, viapts_nbr=k, zeta = zeta_, rho=rho_, lag=lag_, des_vector=des_vector,\
                            rotx_init=rot[0],roty_init=rot[1], rotz_init=rot[2],base = base,equ_paras = equ_paras))
        i+=1
        prob1.driver = pyOptSparseDriver()
        prob1.driver.options['optimizer'] = 'SNOPT'
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
        flag,detection = collision_check(prob1['rot_p'],prob1['d2'],prob1['d4'],prob1['d6'],\
                                    prob1['tube_ends'],num_nodes,mesh,k)
        error = prob1['targetnorm']
        log(count,multiplier_zeta,i,flag,error,detection)
        mdictf = {'points':prob1['integrator_group3.state:p'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                            'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                            'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                            'initial_condition_dpsi':prob1['initial_condition_dpsi'], 'loc':prob1['loc'], 'rotx':prob1['rotx'], 'roty':prob1['roty'],
                            'rotz':prob1['rotz'],'d7':prob1['d7'], 'd8':prob1['d8'],'ee':prob1['desptsconstraints'],'tipvec':prob1['tipvec'],
                            'rho':rho_,'lag':lag_,'zeta':zeta_, 
                            'error':prob1['targetnorm'], 'tip_position':prob1['desptsconstraints'],
                            }
        scipy.io.savemat('siml_x'+str(i)+'.mat',mdictf)
        jointvalues_adrs = 'siml_x'+str(i)+'.mat'
        if error.any() >= 3:
            multiplier_rho = error * 10
            rho_ = multiplier_rho + rho_
            lag_ = lag_ + (rho_) * error/norm1 
        

        if error.any() < 3 and flag == 0 :
            mdict2 = {'points':prob1['integrator_group3.state:p'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                            'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                            'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                            'initial_condition_dpsi':prob1['initial_condition_dpsi'], 'loc':prob1['loc'], 'rotx':prob1['rotx'], 'roty':prob1['roty'],
                            'rotz':prob1['rotz'],'d7':prob1['d7'], 'd8':prob1['d8'],'ee':prob1['desptsconstraints'],'tipvec':prob1['tipvec'],
                            'rho':rho_,'lag':lag_,'zeta':zeta_, 
                            'error':prob1['targetnorm'], 'tip_position':prob1['desptsconstraints'],
                            }
            scipy.io.savemat('siml_f_'+str(i)+'.mat',mdict2)
            break
    

