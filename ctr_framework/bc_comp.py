import numpy as np
from openmdao.api import ExplicitComponent


class BcComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('tube_nbr', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        self.add_input('dpsi_ds',shape=(num_nodes,k,tube_nbr))


        self.add_output('torsionconstraint',shape=((k,tube_nbr)))

        # partials
        self.declare_partials('torsionconstraint','dpsi_ds')
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        dpsi_ds = inputs['dpsi_ds']
        bc = np.zeros((k,tube_nbr))
    
        bc[:,:] = dpsi_ds[-1,:,:]
        
        outputs['torsionconstraint'] = bc
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        dpsi_ds = inputs['dpsi_ds']
        tube_nbr = self.options['tube_nbr']
        
        '''Computing Partials'''
        ppsi_dot = np.zeros((tube_nbr*k, num_nodes*k*tube_nbr))
        ppsi_dot[:,(num_nodes-1)*k*tube_nbr:] = np.identity(k*tube_nbr)
    
        partials['torsionconstraint','dpsi_ds']= ppsi_dot


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes =3
    k = 1
    tube_nbr = 4
  
    tip=np.random.random((k,tube_nbr))*3
    dpsi=np.random.random((num_nodes,k,tube_nbr))
    comp.add_output('tube_ends_tip',val=tip)
    comp.add_output('dpsi_ds',val=dpsi)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = BcComp(k=k,num_nodes=num_nodes,tube_nbr=tube_nbr)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.model.list_outputs()
    
    
