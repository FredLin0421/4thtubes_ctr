import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class SComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining the sin() tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs
        self.add_input('psi', shape=(num_nodes,k,tube_nbr))
        

        # outputs
        self.add_output('S',shape=(num_nodes,k,tube_nbr,tube_nbr))
        
        row_indices_S = np.outer(np.arange(0,num_nodes*k*tube_nbr*tube_nbr),np.ones(tube_nbr))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr**2),np.arange(tube_nbr)).flatten())\
                         + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        self.declare_partials('S', 'psi', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
       

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        psi = inputs['psi']
        
        

        
        
        S = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        s = np.ones((num_nodes,k,tube_nbr))
        for i in range(tube_nbr):
            for z in range(tube_nbr):
                S[:,:,i,z] = np.sin(psi[:,:,i] - psi[:,:,z])
        
        outputs['S'] = S


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        psi = inputs['psi']
        
        Pk_ps = np.zeros((num_nodes*k,tube_nbr**2,tube_nbr))
        Pk_ps[:, 1,0] = np.cos(psi[:,:,0]-psi[:,:,1]).flatten()
        Pk_ps[:, 2,0] = np.cos(psi[:,:,0]-psi[:,:,2]).flatten()
        Pk_ps[:, 3,0] = np.cos(psi[:,:,0]-psi[:,:,3]).flatten()
        Pk_ps[:, 4,0] = -np.cos(psi[:,:,1]-psi[:,:,0]).flatten()
        Pk_ps[:, 8,0] = -np.cos(psi[:,:,2]-psi[:,:,0]).flatten()
        Pk_ps[:, 12,0] = -np.cos(psi[:,:,3]-psi[:,:,0]).flatten()


        Pk_ps[:, 1,1] = -np.cos(psi[:,:,0]-psi[:,:,1]).flatten()
        Pk_ps[:, 4,1] = np.cos(psi[:,:,1]-psi[:,:,0]).flatten()
        Pk_ps[:, 6,1] = np.cos(psi[:,:,1]-psi[:,:,2]).flatten()
        Pk_ps[:, 7,1] = np.cos(psi[:,:,1]-psi[:,:,3]).flatten()
        Pk_ps[:, 9,1] = -np.cos(psi[:,:,2]-psi[:,:,1]).flatten()
        Pk_ps[:, 13,1] = -np.cos(psi[:,:,3]-psi[:,:,1]).flatten()

        Pk_ps[:, 2,2] = -np.cos(psi[:,:,0]-psi[:,:,2]).flatten()
        Pk_ps[:, 6,2] = -np.cos(psi[:,:,1]-psi[:,:,2]).flatten()
        Pk_ps[:, 8,2] = np.cos(psi[:,:,2]-psi[:,:,0]).flatten()
        Pk_ps[:, 9,2] = np.cos(psi[:,:,2]-psi[:,:,1]).flatten()
        Pk_ps[:, 11,2] = np.cos(psi[:,:,2]-psi[:,:,3]).flatten()
        Pk_ps[:, 14,2] = -np.cos(psi[:,:,3]-psi[:,:,2]).flatten()

        Pk_ps[:, 3,3] = -np.cos(psi[:,:,0]-psi[:,:,3]).flatten()
        Pk_ps[:, 7,3] = -np.cos(psi[:,:,1]-psi[:,:,3]).flatten()
        Pk_ps[:, 11,3] = -np.cos(psi[:,:,2]-psi[:,:,3]).flatten()
        Pk_ps[:, 12,3] = np.cos(psi[:,:,3]-psi[:,:,0]).flatten()
        Pk_ps[:, 13,3] = np.cos(psi[:,:,3]-psi[:,:,1]).flatten()
        Pk_ps[:, 14,3] = np.cos(psi[:,:,3]-psi[:,:,2]).flatten()

        partials['S','psi'] = Pk_ps.flatten()
        
        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=15
    k=3
    tube_nbr = 4
    comp = IndepVarComp()
    comp.add_output('psi', val=np.random.random((n,k,tube_nbr)))
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = SComp(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('Scomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
