import numpy as np
from openmdao.api import ExplicitComponent


class InitialpsiComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('tube_nbr', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        self.add_input('initial_condition_dpsi',shape=(k,tube_nbr))
        self.add_input('alpha',shape=(k,tube_nbr))
        self.add_input('beta',shape=(k,tube_nbr))
 
        self.add_output('initial_condition_psi',shape=((k,tube_nbr)))

        # partials
        # define indices
        self.declare_partials('initial_condition_psi','alpha',rows=np.arange(k*tube_nbr) , cols=np.arange(k*tube_nbr))
        self.declare_partials('initial_condition_psi','beta',rows=np.arange(k*tube_nbr) , cols=np.arange(k*tube_nbr))
        self.declare_partials('initial_condition_psi','initial_condition_dpsi')
        
    def compute(self,inputs,outputs):
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        alpha = inputs['alpha']
        beta = inputs['beta']
        initial_condition_dpsi = inputs['initial_condition_dpsi']
        
        init = np.zeros((k,tube_nbr))
        init = alpha - beta * initial_condition_dpsi
        outputs['initial_condition_psi'] = init
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        initial_condition_dpsi = inputs['initial_condition_dpsi']
        beta = inputs['beta']
        alpha = inputs['alpha']
        
        
        '''Computing Partials'''
        Pi_pa = np.ones((k,tube_nbr))
        # Pi_pa[:,0] = 1
        # Pi_pa[:,1] = 1
        # Pi_pa[:,2] = 1
        partials['initial_condition_psi','alpha']= Pi_pa.flatten()

        Pi_pb = np.zeros((k,tube_nbr))
        Pi_pb[:,:] = -initial_condition_dpsi[:,:]
        # Pi_pb[:,1] = -initial_condition_dpsi[:,1]
        # Pi_pb[:,2] = -initial_condition_dpsi[:,2]
        # Pi_pb[:,3] = -initial_condition_dpsi[:,3]
        partials['initial_condition_psi','beta']= Pi_pb.flatten()

        Pi_pdpsi = np.zeros((tube_nbr*k, k*tube_nbr))
        Pi_pdpsi[:tube_nbr*k,:tube_nbr*k] = -np.diag(beta.flatten())
        partials['initial_condition_psi','initial_condition_dpsi']= Pi_pdpsi

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes = 35
    k = 1
    tube_nbr = 4
    dpsi = np.random.random((k,tube_nbr))
    al =  np.random.random((k,tube_nbr))
    be =  np.random.random((k,tube_nbr))

    comp.add_output('alpha',val = al)
    comp.add_output('beta',val = be)
    comp.add_output('initial_condition_dpsi',val=dpsi)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = InitialpsiComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    # prob.check_partials(compact_print=True)
    prob.model.list_outputs()
