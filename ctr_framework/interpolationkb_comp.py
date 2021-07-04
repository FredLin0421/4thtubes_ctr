import numpy as np
from openmdao.api import ExplicitComponent
import math


class InterpolationkbComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('tube_nbr', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs
        self.add_input('tube_ends_hyperbolic',shape=(num_nodes,k,tube_nbr))
        self.add_input('tube_ends_tip',shape=(k,tube_nbr))


        # outputs
        self.add_output('tube_ends',shape=(num_nodes,k,tube_nbr))
        


        # partials
        
        col_indices_b = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr),np.arange(tube_nbr)).flatten())\
                             + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        row_indices_b = np.outer(np.arange(num_nodes*k*tube_nbr),np.ones(tube_nbr)).flatten()
        self.declare_partials('tube_ends','tube_ends_hyperbolic', rows = row_indices_b,cols=col_indices_b.flatten())
        self.declare_partials('tube_ends', 'tube_ends_tip')


    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        tube_ends_hyperbolic = inputs['tube_ends_hyperbolic']
        tube_ends_tip = inputs['tube_ends_tip']
        
        interpolation_idx = np.floor(tube_ends_tip).astype(int)
        interpolation_val = tube_ends_tip - np.floor(tube_ends_tip)
        

        self.interpolate_idx = interpolation_idx
        self.interpolate_val = interpolation_val
        for i in range(tube_nbr):
            tube_ends_hyperbolic[interpolation_idx[:,i],:,i] = interpolation_val[:,i]
        outputs['tube_ends'] = tube_ends_hyperbolic        
        

    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        interpolation_idx = self.interpolate_idx
        """partials Jacobian of partial derivatives."""
        '''tube_ends'''
        
        Pe_pb = np.zeros((num_nodes*k*tube_nbr,k*tube_nbr))
        k_ = np.arange(k)
        # r_idx0 = interpolation_idx[:,0] * k * tube_nbr + k_ * tube_nbr
        # c_idx0 = k_*tube_nbr
        # r_idx1 = interpolation_idx[:,1] * k * tube_nbr + 1 + k_ * tube_nbr
        # c_idx1 = k_*tube_nbr+1
        # r_idx2 = interpolation_idx[:,2] * k * tube_nbr + 2 + k_ * tube_nbr
        # c_idx2 = k_*tube_nbr+2
        # r_idx3 = interpolation_idx[:,3] * k * tube_nbr + 3 + k_ * tube_nbr
        # c_idx3 = k_*tube_nbr+3
        # Pe_pb[r_idx0,c_idx0] = 1
        # Pe_pb[r_idx1,c_idx1] = 1
        # Pe_pb[r_idx2,c_idx2] = 1
        # Pe_pb[r_idx3,c_idx3] = 1
                
        Pt_pb = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        for i in range(tube_nbr):
            Pt_pb[:,:,i,i] = 1
            Pt_pb[interpolation_idx[:,i],:,i,i] = 0
            r_idx = interpolation_idx[:,i] * k * tube_nbr + i + k_ * tube_nbr
            c_idx = k_*tube_nbr + i
            Pe_pb[r_idx,c_idx] = 1
        
        partials['tube_ends','tube_ends_tip'][:] = Pe_pb
        partials['tube_ends','tube_ends_hyperbolic'][:] = Pt_pb.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n = 8
    k = 1
    idx = int(n/2) 
    tube_nbr = 4
    hyper = np.zeros((n,k,tube_nbr))
    hyper[:idx,:,:] = 1
    comp.add_output('tube_ends_hyperbolic',val = hyper )
    beta_init = np.zeros((k,tube_nbr))
    beta_init[:,0] = 3*n/4
    beta_init[:,1] = n/2
    beta_init[:,2] = n/8
    beta_init[:,3] = n/8

    comp.add_output('tube_ends_tip', val=beta_init)
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = InterpolationkbComp(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('interpolationknComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
  


