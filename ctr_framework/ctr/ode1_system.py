import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class ODE1System(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('tube_nbr', default=4, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('num_nodes', default=5, types=int)
        
        

    '''This class is defining the sin() tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs
        self.add_input('psi', shape=(num_nodes,k,tube_nbr))
        self.add_input('K_out', shape=(num_nodes,k,tube_nbr,tube_nbr))
        

        # outputs
        self.add_output('dpsi_ds',shape=(num_nodes,k,tube_nbr))
        
        row_indices_S = np.outer(np.arange(0,num_nodes*k*tube_nbr),np.ones(tube_nbr)).flatten()
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr),np.arange(tube_nbr)).flatten())\
                         + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        col_indices_k = np.arange(num_nodes*k*tube_nbr*tube_nbr)
        self.declare_partials('dpsi_ds', 'K_out',rows=row_indices_S,cols=col_indices_k)
        self.declare_partials('dpsi_ds', 'psi', rows=row_indices_S, cols=col_indices_S.flatten())
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        psi = inputs['psi']
        K_out = inputs['K_out']
        
        
        S = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        for i in range(tube_nbr):
            for z in range(tube_nbr):
                S[:,:,i,z] = np.sin(psi[:,:,i] - psi[:,:,z])
        
        outputs['dpsi_ds'] = np.sum(K_out*S,3)


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        psi = inputs['psi']
        K_out = inputs['K_out']
        
        Pk_ps = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        
        
        
        for i in range(tube_nbr):
            for j in range(tube_nbr):
                Pk_ps[:,:, i,j] = -np.cos(psi[:,:,i]-psi [:,:,j]) * K_out[:,:,i,j]
        Pk_ps[:,:, 0,0] = np.cos(psi[:,:,0]-psi [:,:,1]) * K_out[:,:,0,1] + np.cos(psi[:,:,0]-psi[:,:,2]) * K_out[:,:,0,2] \
                                + np.cos(psi[:,:,0]-psi[:,:,3]) * K_out[:,:,0,3]
        Pk_ps[:,:, 1,1] = np.cos(psi[:,:,1]-psi [:,:,0]) * K_out[:,:,1,0] + np.cos(psi[:,:,1]-psi[:,:,2]) * K_out[:,:,1,2] \
                                + np.cos(psi[:,:,1]-psi[:,:,3]) * K_out[:,:,1,3]
        Pk_ps[:,:, 2,2] = np.cos(psi[:,:,2]-psi [:,:,0]) * K_out[:,:,2,0] + np.cos(psi[:,:,2]-psi[:,:,1]) * K_out[:,:,2,1] \
                                + np.cos(psi[:,:,2]-psi[:,:,3]) * K_out[:,:,2,3]
        Pk_ps[:,:, 3,3] = np.cos(psi[:,:,3]-psi [:,:,0]) * K_out[:,:,3,0] + np.cos(psi[:,:,3]-psi[:,:,1]) * K_out[:,:,3,1] \
                                + np.cos(psi[:,:,3]-psi[:,:,2]) * K_out[:,:,3,2]
            



        Pp_pk = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        for i in range(tube_nbr):
            for j in range(tube_nbr):
                Pp_pk[:,:,i,j] = np.sin(psi[:,:,i] - psi[:,:,j])


        partials['dpsi_ds','psi'] = Pk_ps.flatten()
        partials['dpsi_ds','K_out'] = Pp_pk.flatten()
        
        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=2
    k=1
    tube_nbr = 4
    comp = IndepVarComp()
    comp.add_output('psi', val=np.random.random((n,k,tube_nbr)))
    comp.add_output('K_out', val=np.random.random((n,k,tube_nbr,tube_nbr)))
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ODE1System(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('Scomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)