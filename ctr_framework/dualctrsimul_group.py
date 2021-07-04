import numpy as np
import scipy
import os

from openmdao.api import Problem, Group, ExecComp, IndepVarComp, ScipyOptimizeDriver, pyOptSparseDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None
from ctr_framework.ctrsimul_group import CtrsimulGroup
from ctr_framework.objsmulti_comp import ObjsmultiComp


class DualctrsimulGroup(Group):
    def initialize(self):
        self.options.declare('num_nodes', default=100, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('k_', default=1, types=int)
        self.options.declare('i')
        self.options.declare('a', default=2, types=int)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('pt')
        self.options.declare('tar_vector')
        self.options.declare('rot')
        self.options.declare('base')
        self.options.declare('equ_paras')
        self.options.declare('viapts_nbr')
        self.options.declare('zeta')
        self.options.declare('rho')
        self.options.declare('lag')
        self.options.declare('meshfile')
        self.options.declare('leftpath')
        self.options.declare('rightpath')

        
        

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        k_ = self.options['k_']
        i = self.options['i']
        a = self.options['a']
        pt = self.options['pt']
        equ_paras = self.options['equ_paras']
        tar_vector = self.options['tar_vector']
        tube_nbr = self.options['tube_nbr']
        rot = self.options['rot']
        viapts_nbr = self.options['viapts_nbr']
        base = self.options['base']
        zeta = self.options['zeta']
        rho = self.options['rho']
        lag = self.options['lag']
        meshfile = self.options['meshfile']
        leftpath = self.options['leftpath']
        rightpath = self.options['rightpath']
        
        
        # add subsystem
        'ctr'
        # zeta = np.sum(zeta,axis=0)/3
        # rho = np.sum(rho,axis=0/3)
        # lag = np.sum(lag,axis=0/3)
        #fff1 20,15
        # rho[:,:k-1,:] = rho[:,:k-1,:] * 10
        # rho[:,-1,:] = rho[:,-1,:] * 5
        # zeta[:,:,:] = zeta[:,:,:] * 20
        # zeta[2,:,:] = zeta[2,:,:] * 50
        # heartcase01
        jointvalues1_adrs = 'siml.mat'
        ctr1group = CtrsimulGroup(k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, \
                        pt=pt[0,:,:], meshfile = meshfile, pt_full=pt[0,:,:],\
                        jointvalues_adrs = jointvalues1_adrs,\
                        viapts_nbr=viapts_nbr, zeta = zeta[0,:,:],rho=rho[0,:,:], lag=lag[0,:,:],\
                        rotx_init=rot[0,0],roty_init=rot[0,1],rotz_init=rot[0,2],base = base[0,:].reshape(3,1),equ_paras = equ_paras)
        
        # heartcase03
        jointvalues2_adrs= 'simr.mat'
        ctr2group = CtrsimulGroup(k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, \
                        pt=pt[1,:,:], meshfile=meshfile, pt_full=pt[1,:,:],\
                        jointvalues_adrs=jointvalues2_adrs,\
                        viapts_nbr=viapts_nbr, zeta = zeta[1,:,:], rho=rho[1,:,:], lag=lag[1,:,:],\
                        rotx_init=rot[1,0],roty_init=rot[1,1],rotz_init=rot[1,2],base = base[1,:].reshape(3,1),equ_paras = equ_paras)


        self.add_subsystem('Ctr1Group', ctr1group)
        self.add_subsystem('Ctr2Group', ctr2group)
        
        # collision between two arms

        # objectives
        self.connect('Ctr1Group.objs','objs_a1')
        self.connect('Ctr2Group.objs','objs_a2')
        objsmulticomp = ObjsmultiComp(k=k,num_nodes=num_nodes)
        self.add_subsystem('ObjsmultiComp', objsmulticomp, promotes=['*'])
        self.add_objective('objsmulti')


        
