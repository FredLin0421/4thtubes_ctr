import numpy as np
from openmdao.api import ExplicitComponent


class NormtipvecComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']


        # Inputs
        self.add_input('tipvec',shape=(k,3))
        self.add_input('norm',shape=((k,1)))

        # Outputs
        self.add_output('norm_tipvec', shape=(k,3))

        
        row_indices = np.arange(k*3)
        col_indices = np.outer(np.arange(k),np.ones(3))
    
        self.declare_partials('norm_tipvec', 'tipvec')
        self.declare_partials('norm_tipvec', 'norm',rows = row_indices.flatten(),cols=col_indices.flatten())

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        tipvec = inputs['tipvec']
        norm = inputs['norm']
        norm_tipvec = np.zeros((k,3))
        norm_tipvec[:,:] = tipvec/norm
        
        outputs['norm_tipvec'] = norm_tipvec


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        
        k = self.options['k']
        tipvec = inputs['tipvec']
        norm = inputs['norm']

        '''Computing Partials'''
        pnt_pn = np.zeros((k,3))
        # pnt_pn[:,:] = -tipvec * norm[:,np.newaxis]**-2
        pnt_pn[:,:] = -tipvec * norm**-2
        # pnt_pn[:,1] = 
        # pnt_pn[:,2] =  


        partials['norm_tipvec','tipvec'][:]= np.identity(k*3)
        partials['norm_tipvec','norm'][:]= pnt_pn.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=40
    k=10
    comp = IndepVarComp()
    comp.add_output('tipvec', val=np.random.random((k,3))*10)
    comp.add_output('norm', val=np.ones((k,1)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = NormtipvecComp(num_nodes=n,k=k,tube_nbr=3)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
