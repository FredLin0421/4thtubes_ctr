import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem

class ODE2profile(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=54, types=int)
        self.options.declare('num_nodes', default=10, types=int)

    def setup(self):
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # input
        self.add_input('R',shape=(num_nodes,k,3,3))
        self.add_input('p',shape=(num_nodes,k,3,1))
        # output
        self.add_output('R_',shape=(num_nodes,k,3,3))
        self.add_output('p_',shape=(num_nodes,k,3,1))
        # partials
        val = np.ones(num_nodes*k*3*3)
        rows = np.arange(num_nodes*k*3*3)
        cols = np.arange(num_nodes*k*3*3)
        self.declare_partials('R_', 'R',rows = rows, cols = cols,val=val)
        valp = np.ones(num_nodes*k*3)
        rowsp = np.arange(num_nodes*k*3)
        colsp = np.arange(num_nodes*k*3)
        self.declare_partials('p_', 'p',rows = rowsp, cols = colsp,val=valp)
        
    def compute(self,inputs,outputs):
        
        
        outputs['p_'] = inputs['p'] 
        outputs['R_'] = inputs['R']


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=5
    k=1

    comp.add_output('R', val=np.random.random((n,k,3,3)))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ODE2profile(num_nodes=n,k=k)
    group.add_subsystem('bbpoints', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)