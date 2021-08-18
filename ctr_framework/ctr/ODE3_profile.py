import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem

class ODE3profile(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('num_nodes', default=10, types=int)

    def setup(self):
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # input
        self.add_input('p',shape=(num_nodes,k,3,1))
        
        # output
        self.add_output('profile_output',shape=(num_nodes,k,3,1))
        
        # partials
        val = np.ones(num_nodes*k*3)
        rows = np.arange(num_nodes*k*3)
        cols = np.arange(num_nodes*k*3)
        self.declare_partials('profile_output', 'p',rows = rows, cols = cols,val=val)
        
    def compute(self,inputs,outputs):
        
        p = inputs['p'] 
        outputs['profile_output'] = p 


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=5
    k=1

    comp.add_output('p', val=np.random.random((n,k,3,1)))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ODE3Profile(num_nodes=n,k=k)
    group.add_subsystem('bbpoints', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)