import numpy as np
from openmdao.api import ExplicitComponent


class EndeffectorComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        #Inputs
        self.add_input('tip_trans',shape=(k,4,4))

        # outputs
        self.add_output('Tee',shape=(k,3))
        
        self.declare_partials('Tee', 'tip_trans')

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tip_trans = inputs['tip_trans']
        
        T = np.zeros((k,4,4))
        idx_ = np.arange(3)
        T[:,idx_,idx_] = 1
        T[:,3,3] = 1
        T[:,2,3] = 5
        ee = T @ tip_trans   
        outputs['Tee'] = ee[:,:3,3] 


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        '''Computing Partials'''
        
        pe_pt = np.zeros((k*3,k*4*4))
        col_ = np.tile(np.array([3,7,11]),k) + 16*np.outer(np.arange(k),np.ones(3)).flatten()
        row_ = np.arange(k*3)
        
        pe_pt[row_,col_.astype(int)] = 1
        col_1 = np.tile(np.array([15]),k) + 16*np.outer(np.arange(k),np.ones(1)).flatten()
        row_1 = np.arange(2,k*3,3)
        pe_pt[row_1,col_1.astype(int)] = 5



        partials['Tee','tip_trans'][:] = pe_pt
        

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=4
    k=3
    comp = IndepVarComp()
    comp.add_output('tip_trans', val=np.random.random((k,4,4)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = EndeffectorComp(num_nodes=n,k=k,tube_nbr=4)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
