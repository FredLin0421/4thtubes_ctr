import numpy as np
from openmdao.api import ExplicitComponent


class InitialdpsiComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('tube_nbr', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        self.add_input('dpsi_ds',shape=(num_nodes,k,tube_nbr))


        self.add_output('initial_condition_dpsi',shape=(k,tube_nbr))

        # partials
        # define indices
        
        self.declare_partials('initial_condition_dpsi','dpsi_ds')
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        dpsi_ds = inputs['dpsi_ds']
        outputs['initial_condition_dpsi'] = dpsi_ds[0,:,:]
        
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        dpsi_ds = inputs['dpsi_ds']
       
        
        '''Computing Partials'''

        Pi_pdpsi = np.zeros((tube_nbr*k, num_nodes*k*tube_nbr))
        Pi_pdpsi[:tube_nbr*k,:tube_nbr*k] = np.diag(np.ones((tube_nbr*k)))
        partials['initial_condition_dpsi','dpsi_ds']= Pi_pdpsi
       
        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes = 10
    k = 3
    tube_nbr = 4
    dpsi = np.random.random((num_nodes,k,tube_nbr))
    

    
    comp.add_output('dpsi_ds',val=dpsi)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = InitialdpsiComp(tube_nbr = tube_nbr,num_nodes=num_nodes)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    prob.model.list_outputs()
