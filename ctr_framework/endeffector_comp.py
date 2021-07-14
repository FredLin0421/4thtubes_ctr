import numpy as np
from openmdao.api import ExplicitComponent


class EndeffectorComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)
        self.options.declare('ee_length', default=5, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        #Inputs
        self.add_input('tip_trans',shape=(k,4,4))

        # outputs
        self.add_output('Tee',shape=(k,4,4))
        
        self.declare_partials('Tee', 'tip_trans')

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        ee_length = self.options['ee_length']
        num_nodes= self.options['num_nodes']
        tip_trans = inputs['tip_trans']
        
        T = np.zeros((k,4,4))
        idx_ = np.arange(3)
        T[:,idx_,idx_] = 1
        T[:,3,3] = 1
        T[:,2,3] = ee_length
        'ee = T @ tip_trans'
        ee = tip_trans @ T 
        outputs['Tee'] = ee 


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        ee_length = self.options['ee_length']
        k = self.options['k']
        '''Computing Partials'''
        
        add = np.arange(k) * 16
        col= np.outer(np.ones(k),[2,6,10,14]) + add[:,np.newaxis]
        row = np.outer(np.ones(k),[3,7,11,15]) + add[:,np.newaxis]
        pt_pt = np.identity((k*4*4))
        pt_pt[row.astype(int),col.astype(int)] = ee_length


        partials['Tee','tip_trans'][:] = pt_pt
        

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=4
    k=1
    comp = IndepVarComp()
    comp.add_output('tip_trans', val=np.random.random((k,4,4)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = EndeffectorComp(num_nodes=n,k=k,tube_nbr=4,ee_length=5)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
