import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem

class KinematicsComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=40, types=int)

    def setup(self):
        
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # inputs
        
        self.add_input('RHS',shape=(num_nodes,k,tube_nbr))
        self.add_input('dpsi_ds',shape=(num_nodes,k,tube_nbr))

        # output
        self.add_output('psi_dot',shape=(num_nodes,k,tube_nbr))
        self.add_output('dpsi_ds_dot',shape=(num_nodes,k,tube_nbr))


        # partials
        
        col_indices_RHS = np.arange(num_nodes*k*tube_nbr).flatten()
        row_indices_RHS = np.arange(num_nodes*k*tube_nbr).flatten()
        col_indices_dpsi = np.arange(num_nodes*k*tube_nbr).flatten()
        row_indices_dpsi = np.arange(num_nodes*k*tube_nbr).flatten()
        self.declare_partials('psi_dot', 'dpsi_ds',rows=row_indices_dpsi,cols=col_indices_dpsi)
        self.declare_partials('dpsi_ds_dot','RHS',rows=row_indices_RHS,cols=col_indices_RHS)

    def compute(self,inputs,outputs):
        
        
        RHS = inputs['RHS']
        dpsi_ds = inputs['dpsi_ds']
        "Compute 2 first order equations"
        outputs['psi_dot'] = dpsi_ds
        outputs['dpsi_ds_dot'] = RHS  

  
        
    def compute_partials(self, inputs, partials):
        
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        partials['psi_dot','dpsi_ds'][:] = np.ones((num_nodes*k*tube_nbr)).flatten()
        partials['dpsi_ds_dot','RHS'][:] = np.ones((num_nodes*k*tube_nbr)).flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    n = 17
    tube_nbr = 4
    k=1
    comp = IndepVarComp()
    comp.add_output('RHS', val=np.ones((n,k,tube_nbr)))
    comp.add_output('dpsi_ds', val=np.ones((n,k,tube_nbr)))
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KinematicsComp(num_nodes=n,k=k,tube_nbr = tube_nbr)
    group.add_subsystem('kinematicscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)