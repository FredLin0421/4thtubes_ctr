import numpy as np
from openmdao.api import ExplicitComponent
import math


class InterpolationkpComp(ExplicitComponent):

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
        self.add_input('straight_ends_hyperbolic',shape=(num_nodes,k,tube_nbr))
        self.add_input('straight_ends_tip',shape=(k,tube_nbr))
        self.add_input('kappa',shape=(1,tube_nbr))


        # outputs
        self.add_output('straight_ends',shape=(num_nodes,k,tube_nbr))
        
        # partials
        
        col_indices_b = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr),np.arange(tube_nbr)).flatten())\
                             + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        row_indices_b = np.outer(np.arange(num_nodes*k*tube_nbr),np.ones(tube_nbr)).flatten()
        self.declare_partials('straight_ends','straight_ends_hyperbolic', rows = row_indices_b,cols=col_indices_b.flatten())
        self.declare_partials('straight_ends', 'straight_ends_tip')
        self.declare_partials('straight_ends', 'kappa')

    def compute(self,inputs,outputs):
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        straight_ends_hyperbolic = inputs['straight_ends_hyperbolic']
        straight_ends_tip = inputs['straight_ends_tip']
        kappa = inputs['kappa']
        interpolation_idx = np.floor(straight_ends_tip).astype(int)
        interpolation_val = straight_ends_tip - np.floor(straight_ends_tip)
        self.interpolate_idx = interpolation_idx
        self.interpolate_val = interpolation_val
        for i in range(tube_nbr):
            straight_ends_hyperbolic[interpolation_idx[:,i],:,i] = interpolation_val[:,i] * kappa[:,i]
        
        outputs['straight_ends'] = straight_ends_hyperbolic        


    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        interpolation_idx = self.interpolate_idx
        interpolation_val = self.interpolate_val
        """partials Jacobian of partial derivatives."""

        Ps_pkp = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        Pe_pb = np.zeros((num_nodes*k*tube_nbr,k*tube_nbr))
        Pt_pb = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        k_ = np.arange(k)
        for i in range(tube_nbr):
            Ps_pkp[interpolation_idx[:,i],:,i,i] = interpolation_val[:,i]
            Pt_pb[:,:,i,i] = 1
            Pt_pb[interpolation_idx[:,i],:,i,i] = 0

        Pe_pb[:k*tube_nbr,:k*tube_nbr] = np.identity(k*tube_nbr)
        partials['straight_ends','kappa'][:] = Ps_pkp.reshape((num_nodes*k*tube_nbr,tube_nbr))
        partials['straight_ends','straight_ends_tip'][:] = Pe_pb
        partials['straight_ends','straight_ends_hyperbolic'][:] = Pt_pb.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n = 17
    k = 4
    tube_nbr = 4
    idx = int(n/3) 
    hyper = np.zeros((n,k,tube_nbr))
    hyper[:idx,:,:] = 1
    hyper = np.random.random((n,k,tube_nbr))
    comp.add_output('straight_ends_hyperbolic',val = hyper )
    beta_init = np.zeros((k,tube_nbr))
    beta_init[:,0] = 3*n/4
    beta_init[:,1] = n/2
    beta_init[:,2] = n/8
    beta_init[:,3] = n/8
    beta_init = np.random.random((k,tube_nbr))
    comp.add_output('straight_ends_tip', val=beta_init)
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = InterpolationkpComp(num_nodes=n,k=k,tube_nbr = tube_nbr)
    group.add_subsystem('interpolationknComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
  


