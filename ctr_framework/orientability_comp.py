import numpy as np
from openmdao.api import ExplicitComponent

class OrientabilityComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)
        self.options.declare('des_vector')
        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('norm_tipvec',shape=(k,3))

        # outputs
        # self.add_output('orientability',shape=(k))
        self.add_output('orientability')
        
        row_indices = np.outer(np.arange(k),np.ones(3)).flatten()
        col_indices = np.arange(k*3)
        
        self.declare_partials('orientability', 'norm_tipvec')#,rows=row_indices,cols=col_indices)
        

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        des_vector = self.options['des_vector']
        norm_tipvec = inputs['norm_tipvec']
        
        norm = np.linalg.norm(des_vector - norm_tipvec,axis=1)
        # norm = (des_vector-norm_tipvec)**2
        outputs['orientability'] = np.sum(norm)


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        des_vector = self.options['des_vector']
        norm_tipvec = inputs['norm_tipvec']
        
        '''Computing Partials'''
        sum_ =  des_vector-norm_tipvec
        sumdpsi = np.sum((sum_)**2,1)
        pt_pt = np.zeros((k,3))
        pt_pt[:,0] = -(((sumdpsi)**-0.5) * sum_[:,0])
        pt_pt[:,1] = -(((sumdpsi)**-0.5) * sum_[:,1])
        pt_pt[:,2] = -(((sumdpsi)**-0.5) * sum_[:,2])
        # pt_pt[:,0] = -2 * sum_[:,0]
        # pt_pt[:,1] = -2 * sum_[:,1]
        # pt_pt[:,2] = -2 * sum_[:,2]
        # partials['orientability','norm_tipvec'][:] = pt_pt.flatten()
        partials['orientability','norm_tipvec'][:] = pt_pt.reshape(1,-1)

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=1
    k=3
    tar_vector = np.random.rand(k,3)
    comp = IndepVarComp()
    comp.add_output('norm_tipvec', val=np.random.random((k,3)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = OrientabilityComp(num_nodes=n,k=k,des_vector=tar_vector)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
