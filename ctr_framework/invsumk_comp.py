from __future__ import division
from six.moves import range

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from openmdao.api import ImplicitComponent


class InvsumkComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=50, types=int)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=50, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        # Inputs
        self.add_input('sumkm',shape=(num_nodes,k,tube_nbr,tube_nbr))
        self.add_input('K',shape=(num_nodes,k,tube_nbr,tube_nbr))
        # Outputs
        self.add_output('K_s',shape=(num_nodes,k,tube_nbr,tube_nbr))
        

        
        row_indices = np.arange(num_nodes*k*tube_nbr*tube_nbr) 
        col_indices = np.arange(num_nodes*k*tube_nbr*tube_nbr)
        self.declare_partials('K_s', 'K_s', rows= row_indices, cols= col_indices)
        self.declare_partials('K_s', 'sumkm',rows= row_indices, cols= col_indices)
    
        self.declare_partials('K_s', 'K')
        

    def apply_nonlinear(self, inputs, outputs, residuals):
    
        sumkm = inputs['sumkm']
        K = inputs['K']
        K_s = outputs['K_s']
        residuals['K_s']= sumkm * K_s - K

    def solve_nonlinear(self, inputs, outputs):
        num_nodes = self.options['num_nodes']

        K = inputs['K']
        sumkm = inputs['sumkm']
        
        outputs['K_s'] = K * np.linalg.pinv(sumkm)
        
    def linearize(self, inputs, outputs, jacobian):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k= self.options['k']
        K_s = outputs['K_s']
        K = inputs['K']
        sumkm = inputs['sumkm']

        jacobian['K_s', 'K'][:] = -np.identity(num_nodes*k*tube_nbr*tube_nbr)

        Pu_psk = np.zeros((num_nodes,k,tube_nbr,tube_nbr))

        Pu_pu = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        for i in range(tube_nbr):
            for j in range(tube_nbr):
                Pu_psk[:,:,i,j] = K_s[:,:,i,j]
                Pu_pu[:,:,i,j] = sumkm[:,:,i,j]

        jacobian['K_s', 'K_s'][:] = Pu_pu.flatten()
        jacobian['K_s', 'sumkm'][:] = Pu_psk.flatten()
        self.inv_jac = np.linalg.pinv(np.reshape(Pu_pu,(num_nodes,k,tube_nbr,tube_nbr)))
        


    def solve_linear(self, K_s_outputs, K_s_residuals, mode):
        
        if mode == 'fwd':
            K_s_outputs['K_s'] = self.inv_jac * K_s_residuals['K_s']
        else:
            K_s_residuals['K_s'] = self.inv_jac *  K_s_residuals['K_s']
         
            
if __name__ == '__main__':

  from openmdao.api import Problem, Group
  from openmdao.api import IndepVarComp


  group = Group()

  comp = IndepVarComp()
  num_nodes = 10
  k = 3
  tube_nbr = 4
  
  comp.add_output('K',val=np.random.random((num_nodes,k,tube_nbr,tube_nbr)))
  comp.add_output('sumkm',val=np.random.random((num_nodes,k,tube_nbr,tube_nbr)))
  group.add_subsystem('Inputcomp', comp, promotes=['*'])



  comp = InvsumkComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)

  group.add_subsystem('ucomp', comp, promotes=['*'])


  prob = Problem()
  prob.model = group
  prob.setup()
  prob.run_model()
#   prob.model.list_outputs()
#   prob.check_partials(compact_print=False)
  prob.check_partials(compact_print=True)