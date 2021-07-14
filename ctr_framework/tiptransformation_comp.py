import numpy as np
from openmdao.api import ExplicitComponent


class TiptransformationComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        #Inputs
        self.add_input('tippos',shape=(k,3,1))
        self.add_input('tipori',shape=(k,3,3))
        # outputs
        self.add_output('tip_trans',shape=(k,4,4))
        col_indices1 = np.arange(k*3)
        row_indices1 = np.tile(np.array([3,7,11]),k) + 16*np.outer(np.arange(k),np.ones(3)).flatten()
        col_indices2 = np.arange(k*3*3)
        row_indices2 = np.tile(np.array([0,1,2,4,5,6,8,9,10]),k) + 16*np.outer(np.arange(k),np.ones(3*3)).flatten()

        self.declare_partials('tip_trans', 'tippos',rows = row_indices1.flatten(),cols=col_indices1.flatten())
        self.declare_partials('tip_trans', 'tipori',rows = row_indices2.flatten(),cols=col_indices2.flatten())

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tippos = inputs['tippos']
        tipori = inputs['tipori']
        
        T_pose = np.zeros((k,4,4))
        T_pose[:,:3,:3] = tipori
        T_pose[:,:3,3] = tippos.squeeze()
        T_pose[:,3,3] = 1
        outputs['tip_trans'] = T_pose


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        
        '''Computing Partials'''
        pd_pp = np.zeros((k*3,num_nodes*k*3))
        
        
        pd_pt = np.zeros((k,tube_nbr*3))
        
        partials['tip_trans','tippos'][:] = 1
        partials['tip_trans','tipori'][:]= 1

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=4
    k=10
    comp = IndepVarComp()
    comp.add_output('tippos', val=np.random.random((k,3,1)))
    comp.add_output('tipori', val=np.random.random((k,3,3)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TiptransformationComp(num_nodes=n,k=k,tube_nbr=3)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
