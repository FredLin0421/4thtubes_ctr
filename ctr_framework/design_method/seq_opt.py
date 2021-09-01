import numpy as np
import scipy
import os
import shutil
import time
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
from ctr_framework.equofplane import equofplane
from ctr_framework.findcircle import findCircle

def seq_opt(num_nodes,viapts_nbr,orient_nbr,base,rot,meshfile,pathfile,j):

    #new line
    pts_nbr = viapts_nbr + orient_nbr
    

    tube_nbr = 4
    k=1
    a = 30
    pt = initialize_pt(viapts_nbr,pathfile)
    # pt_pri =  initialize_pt(viapts_nbr * 2,pathfile)
    pt_full =  initialize_pt(viapts_nbr,pathfile)
    p_plane = np.zeros((3,3))
    equ_paras = equofplane(p_plane[0,:],p_plane[1,:],p_plane[2,:])
    norm1 = np.linalg.norm(pt[0,:]-pt[-1,:],ord=1.125)
    center = findCircle(pt[0,1],pt[0,2], \
            pt[-1,1],pt[-1,2],pt[int(viapts_nbr/2),1],pt[int(viapts_nbr/2),2])
    mesh = trianglemesh(num_nodes,k,pt[-1,:],center,meshfile)

    zeta = 0
    rho = 1
    eps_e = 1
    eps_r = 1
    eps_p = 1
    lag = 1
    tol = np.ones((pts_nbr))*2.5
    tol[viapts_nbr:] = 2.5
    t0 = time.time()

    # add reachable points ----- testing
    # pt_reach = np.zeros((pts_nbr,3))
    # pt_reach[:viapts_nbr,:] = pt
    # pt_reach[viapts_nbr:,:] = np.array([[0,-5,182.5],[0,0,185],[0,5,182.5]])  
    
    #### feasibility test
    #pt[-1,:] = np.array([0,0,185])
    '''pt_o = np.zeros((3,3))
    pt_o[:,:] = np.array([-3,0,176])
    pt = np.concatenate((pt,pt_o))'''
    # pt = np.zeros((orient_nbr,3))
    # pt[0,:] = np.array([-3,5,182.5])
    # pt[1,:] = np.array([-3,0,185])
    # pt[2,:] = np.array([-3,-5,182.5])
    # pt[3,:] = np.array([-10,0,182.5])
    # pt[4,:] = np.array([5,0,182.5])

    for i in range(0,viapts_nbr,1):
        count = 1
        count1 = 1
        count_error=1
        trigger = 0
        flag = 1
        error = 1
        lag = 1
        rho = 1 # test
        zeta = 0 # test
        while flag==1 or error>tol[i]:
                
            #option 2
            if flag==1 and count1>=1:
                multiplier = 20*count1
                    # zeta = (1e-3) * multiplier1
                # zeta = (1e-2) * multiplier + zeta
                zeta = (5e-2) * multiplier + zeta
                count1+=1
            prob1 = Problem(model=CtrseqGroup(k=1, num_nodes=num_nodes, a=a, tube_nbr=tube_nbr,\
                    pt=pt[i,:],i=i,target = pt[-1,:], center=center, lag = lag,\
                        zeta=zeta,rho=rho,eps_r=eps_r,eps_p=eps_p, eps_e=eps_e,\
                            pt_full = pt, viapts_nbr=viapts_nbr, meshfile = meshfile,\
                                rotx_init=rot[0],roty_init=rot[1],rotz_init=rot[2],base = base,count=0,equ_paras = equ_paras,pt_test = pt[-1,:]))
            prob1.driver = pyOptSparseDriver()
            prob1.driver.options['optimizer'] = 'SNOPT'
            prob1.driver.opt_settings['Verify level'] = 0
            prob1.driver.opt_settings['Major iterations limit'] = 25
            prob1.driver.opt_settings['Minor iterations limit'] = 1000
            prob1.driver.opt_settings['Iterations limit'] = 1000000
            prob1.driver.opt_settings['Major step limit'] = 2.0
            prob1.driver.opt_settings['Major feasibility tolerance'] = 1.0e-4
            prob1.driver.opt_settings['Major optimality tolerance'] = 1.0e-3
            prob1.driver.opt_settings['Minor feasibility tolerance'] = 1.0e-4
            prob1.setup()
            # init_guess = scipy.io.loadmat('init_2.mat')
            # prob1.set_val('d1', val=init_guess['d1'])
            # prob1.set_val('d2', val=init_guess['d2'])
            # prob1.set_val('d3', val=init_guess['d3'])
            # prob1.set_val('d4', val=init_guess['d4'])
            # prob1.set_val('d5', val=init_guess['d5'])
            # prob1.set_val('d6', val=init_guess['d6'])
            # prob1.set_val('d7', val=init_guess['d7'])
            # prob1.set_val('d8', val=init_guess['d8'])
            # prob1.set_val('kappa', shape=(1,tube_nbr), val=init_guess['kappa'])
            # prob1.set_val('tube_section_length',shape=(1,tube_nbr),val=init_guess['tube_section_length'].T)
            # prob1.set_val('tube_section_straight',shape=(1,tube_nbr),val=init_guess['tube_section_straight'].T)
            # prob1.set_val('alpha', shape=(k,tube_nbr),val=init_guess['alpha'])
            # prob1.set_val('beta', shape=(k,tube_nbr),val=init_guess['beta'])
            # # prob1.set_val('initial_condition_dpsi', shape=(k,tube_nbr), val=init_guess['initial_condition_dpsi'])
            # prob1.set_val('rotx',val=init_guess['rotx'])
            # prob1.set_val('roty',val=init_guess['roty'])
            # prob1.set_val('rotz',val=init_guess['rotz'])
            # prob1.set_val('loc',shape=(3,1),val=init_guess['loc']+1e-10)
            # coefs = np.zeros(num_nodes+1)
            # coefs[num_nodes] = 1.
            # # prob1.set_val('initial_condition_p', val=np.zeros((k,3,1)))
            # prob1.set_val('initial_condition_dpsi',shape=(k,tube_nbr), val=np.random.rand(k,tube_nbr))
            prob1.run()
            prob1.run_model()
            prob1.run_driver()
            # flag,detection = collision_check(prob1['rot_p'],prob1['d2'],prob1['d4'],prob1['d6'],\
            #                         prob1['tube_ends'],num_nodes,mesh,k)
            '''error = prob1['targetnorm']
            log(count,multiplier,i,flag,error,detection)
            if error >= tol[i] or flag==1:
                lag = lag + rho * prob1['targetnorm']/norm1
                rho = count_error * 5
                count_error+=1
                

            elif error<tol[i] and flag==0:             
                break
            trigger = 1
            count+=1'''
            mdict1 = {'points':prob1['p_'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                        'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                        'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                        'd7':prob1['d7'], 'd8':prob1['d8'] ,#'ee':prob1['desptsconstraints'],
                        # 'initial_condition_dpsi':prob1['initial_condition_dpsi'],'rotx':prob1['rotx'],'roty':prob1['roty'], 'rotz':prob1['rotz'],
                        # 'loc':prob1['loc'],'rot_p':prob1['rot_p'],'flag':flag, 'detection':detection, 'zeta':zeta, 'dl0':prob1['tube_section_length'] + prob1['beta'],
                        # 'rho':rho, 'eps_r':eps_r, 'eps_p':eps_p, 'eps_e':eps_e, 
                        # 'lag':lag,
                        }
            scipy.io.savemat('seq_pre'+str(count)+'.mat',mdict1)

        mdict1 = {'points':prob1['p_'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                        'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                        'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                        'd7':prob1['d7'], 'd8':prob1['d8'],#'ee':prob1['desptsconstraints'],
                        # 'initial_condition_dpsi':prob1['initial_condition_dpsi'],'rotx':prob1['rotx'],'roty':prob1['roty'], 'rotz':prob1['rotz'],
                        # 'loc':prob1['loc'],'rot_p':prob1['rot_p'],'flag':flag, 'detection':detection, 'zeta':zeta, 'dl0':prob1['tube_section_length'] + prob1['beta'],
                        # 'rho':rho, 'eps_r':eps_r, 'eps_p':eps_p, 'eps_e':eps_e, 
                        # 'lag':lag
                        }
        scipy.io.savemat('oz2_'+str(i)+'.mat',mdict1)
        os.rename('SNOPT_print.out','SNOPT_print'+str(i)+'.out')

    t1 = time.time()
    print(t1-t0)