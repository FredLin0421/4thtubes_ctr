import numpy as np
from openmdao.api import ExplicitComponent


class KappaComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs
        self.add_input('straight_ends', shape=(num_nodes,k,tube_nbr))
        

        # outputs
        self.add_output('K_kp',shape=(num_nodes,k,tube_nbr,tube_nbr))
        # partials

        row_indices_S = np.outer(np.arange(0,num_nodes*k*tube_nbr*tube_nbr),np.ones(tube_nbr))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr**2),np.arange(tube_nbr)).flatten()) \
                            + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        self.declare_partials('K_kp', 'straight_ends', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        straight_ends = inputs['straight_ends']
        

        K_kp = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        for i in range(tube_nbr):
            for j in range(tube_nbr):
                K_kp[:,:,i,j] = straight_ends[:,:,i] * straight_ends[:,:,j]

        outputs['K_kp'] = K_kp
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        straight_ends = inputs['straight_ends']


        
        Pkkp_ps = np.zeros((num_nodes,k,tube_nbr**2,tube_nbr))
        Pkkp_ps[:,:,0,0] = 2 * straight_ends[:,:,0]
        Pkkp_ps[:,:,1,0] = straight_ends[:,:,1]
        Pkkp_ps[:,:,2,0] = straight_ends[:,:,2]
        Pkkp_ps[:,:,3,0] = straight_ends[:,:,3]     
        Pkkp_ps[:,:,4,0] = straight_ends[:,:,1]
        Pkkp_ps[:,:,8,0] = straight_ends[:,:,2]
        Pkkp_ps[:,:,12,0] = straight_ends[:,:,3]
        

        
        Pkkp_ps[:,:,1,1] = straight_ends[:,:,0]
        Pkkp_ps[:,:,4,1] = straight_ends[:,:,0]
        Pkkp_ps[:,:,5,1] = 2 * straight_ends[:,:,1]
        Pkkp_ps[:,:,6,1] = straight_ends[:,:,2]
        Pkkp_ps[:,:,7,1] = straight_ends[:,:,3]
        Pkkp_ps[:,:,9,1] = straight_ends[:,:,2]
        Pkkp_ps[:,:,13,1] = straight_ends[:,:,3]

        Pkkp_ps[:,:,2,2] = straight_ends[:,:,0]
        Pkkp_ps[:,:,6,2] = straight_ends[:,:,1]
        Pkkp_ps[:,:,8,2] = straight_ends[:,:,0]
        Pkkp_ps[:,:,9,2] = straight_ends[:,:,1]
        Pkkp_ps[:,:,10,2] = 2 * straight_ends[:,:,2]
        Pkkp_ps[:,:,11,2] = straight_ends[:,:,3]
        Pkkp_ps[:,:,14,2] = straight_ends[:,:,3]

        Pkkp_ps[:,:,3,3] = straight_ends[:,:,0]
        Pkkp_ps[:,:,7,3] = straight_ends[:,:,1]
        Pkkp_ps[:,:,11,3] = straight_ends[:,:,2]
        Pkkp_ps[:,:,12,3] = straight_ends[:,:,0]
        Pkkp_ps[:,:,13,3] = straight_ends[:,:,1]
        Pkkp_ps[:,:,14,3] = straight_ends[:,:,2]
        Pkkp_ps[:,:,15,3] = 2 * straight_ends[:,:,3]

        partials['K_kp','straight_ends'][:] =  Pkkp_ps.flatten()



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 175
    k = 1
    tube_nbr = 4
    comp = IndepVarComp()
   
    comp.add_output('straight_ends', val=np.random.random((n,k,tube_nbr))*10)
    
    

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KappaComp(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('kappascomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    