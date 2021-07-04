import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem


class RHSComp(ExplicitComponent):

    def initialize(self):
        
        
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=40, types=int)


    def setup(self):
        # Inputs
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        
        self.add_input('K_out',shape=(num_nodes,k,tube_nbr,tube_nbr))
        self.add_input('S',shape=(num_nodes,k,tube_nbr,tube_nbr))

        # outputs
        self.add_output('RHS', shape=(num_nodes,k,tube_nbr))
        
        # partials
        row_incice_K = np.outer(np.arange(num_nodes*k*tube_nbr),np.ones(tube_nbr)).flatten()
        col_indices_K = np.arange(num_nodes*k*tube_nbr*tube_nbr).flatten()
        self.declare_partials('RHS','K_out',rows=row_incice_K,cols=col_indices_K)
        self.declare_partials('RHS','S',rows=row_incice_K,cols=col_indices_K)

    def compute(self, inputs, outputs):
        
        K = inputs['K_out']        
        S = inputs['S']
        RHS = np.sum(K*S,3)
        
        outputs['RHS'] = RHS
        
        
        
        
        
        


    def compute_partials(self, inputs, partials):
        
        K = inputs['K_out']
        S = inputs['S']
        
        partials['RHS','K_out'][:] = S.flatten()
        partials['RHS','S'][:] = K.flatten()



if __name__ == '__main__':
  from openmdao.api import Problem, Group

  from openmdao.api import IndepVarComp

  group = Group()

  comp = IndepVarComp()
  num_nodes = 30
  k=1
  tube_nbr = 4
  comp.add_output('K_out', val = np.random.random((num_nodes,k,tube_nbr,tube_nbr)))
  comp.add_output('S', val = np.random.random((num_nodes,k,tube_nbr,tube_nbr)))

  group.add_subsystem('comp1', comp, promotes=['*'])

  comp = RHSComp(num_nodes=num_nodes,k=k,tube_nbr = tube_nbr)

  group.add_subsystem('comp2', comp, promotes=['*'])

  prob = Problem()
  prob.model = group
  prob.setup()
  prob.run_model()
  prob.model.list_outputs()

  prob.check_partials(compact_print=True)
#   prob.check_partials(compact_print=False)