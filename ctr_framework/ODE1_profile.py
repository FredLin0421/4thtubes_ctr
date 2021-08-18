import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem

class ODE1profile(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=4, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('num_nodes', default=10, types=int)

    def setup(self):
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # input
        self.add_input('psi',shape=(num_nodes,k,tube_nbr))
        self.add_input('dpsi_ds',shape=(num_nodes,k,tube_nbr))

        # output
        self.add_output('psi_',shape=(num_nodes,k,tube_nbr))
        self.add_output('torsionconstraint',shape=(k,tube_nbr))
        
        # partials
        val = np.ones(num_nodes*k*tube_nbr)
        rows = np.arange(num_nodes*k*tube_nbr)
        cols = np.arange(num_nodes*k*tube_nbr)
        self.declare_partials('psi_', 'psi',rows = rows, cols = cols,val=val)
        rows_t = np.arange(k*tube_nbr) 
        cols_t = np.arange((num_nodes-1)*k*tube_nbr,(num_nodes*k*tube_nbr))
        val_t = np.ones(k*tube_nbr)
        self.declare_partials('torsionconstraint','dpsi_ds',rows=rows_t,cols=cols_t,val=val_t)
    def compute(self,inputs,outputs):
        
        outputs['psi_'] = inputs['psi']
        outputs['torsionconstraint'] = inputs['dpsi_ds'][-1,:,:]



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=20
    k=5
    tube_nbr = 4
    comp.add_output('psi', val=np.random.random((n,k,tube_nbr)))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ODE1profile(num_nodes=n,k=k)
    group.add_subsystem('bbpoints', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    # prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)