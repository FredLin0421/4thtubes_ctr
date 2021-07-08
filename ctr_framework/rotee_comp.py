import numpy as np
from openmdao.api import ExplicitComponent


class RoteeComp(ExplicitComponent):

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
        self.add_input('T',shape=(4,4))

        # outputs
        self.add_output('rotee',shape=(k,3))
        
        row_indices1 = np.outer(np.arange(k*3),np.ones(4)).flatten()
        col_indices1 = np.tile(np.arange(4*3),k).flatten()
        add = np.arange(k) * 16
        col_indices2 = np.outer(np.ones(k),np.tile([3,7,11,15],3)) + add[:,np.newaxis]
        self.declare_partials('rotee', 'tip_trans',rows=row_indices1,cols=col_indices2.flatten())
        self.declare_partials('rotee', 'T',rows=row_indices1,cols=col_indices1)

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tip_trans = inputs['tip_trans']
        T = inputs['T']
        
        
        ee = T @ tip_trans
        outputs['rotee'] = ee[:,:3,3] 


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        tip_trans = inputs['tip_trans']
        T = inputs['T']
        k = self.options['k']
        '''Computing Partials'''
        
        pe_pt = np.zeros((k,3,4))
        pe_pt[:,0,:] = tip_trans[:,:,3]
        pe_pt[:,1,:] = tip_trans[:,:,3]
        pe_pt[:,2,:] = tip_trans[:,:,3]

        pt_pt = np.zeros((k,3,4))
        pt_pt[:,0,:] = T[0,:]
        pt_pt[:,1,:] = T[1,:]
        pt_pt[:,2,:] = T[2,:]
        


        partials['rotee','tip_trans'][:] = pt_pt.flatten()
        partials['rotee','T'][:] = pe_pt.flatten()
        

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=4
    k=10
    comp = IndepVarComp()
    comp.add_output('tip_trans', val=np.random.random((k,4,4)))
    comp.add_output('T', val=np.random.random((4,4)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = RoteeComp(num_nodes=n,k=k,tube_nbr=4)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
