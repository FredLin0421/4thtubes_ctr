import numpy as np
from openmdao.api import ExplicitComponent,Group


class ODE1system(ExplicitComponent):

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
        self.add_input('dpsi_ds',shape=(num_nodes,k,tube_nbr))
        

        # outputs
        self.add_output('psi_dot',shape=(num_nodes,k,tube_nbr))
        self.add_output('psi_ddot',shape=(num_nodes,k,tube_nbr))
        
        row_indices_S = np.outer(np.arange(0,num_nodes*k*tube_nbr),np.ones(tube_nbr)).flatten()
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr),np.arange(tube_nbr)).flatten())\
                         + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        col_indices_k = np.arange(num_nodes*k*tube_nbr*tube_nbr)
        self.declare_partials('psi_ddot', 'K_out',rows=row_indices_S,cols=col_indices_k)
        self.declare_partials('psi_ddot', 'psi', rows=row_indices_S, cols=col_indices_S.flatten())
        col_indices_dpsi = np.arange(num_nodes*k*tube_nbr).flatten()
        row_indices_dpsi = np.arange(num_nodes*k*tube_nbr).flatten()
        self.declare_partials('psi_dot', 'dpsi_ds',rows=row_indices_dpsi,cols=col_indices_dpsi)
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        psi = inputs['psi']
        dpsi_ds = inputs['dpsi_ds']
        K_out = inputs['K_out']
        
        
        S = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        for i in range(tube_nbr):
            for z in range(tube_nbr):
                S[:,:,i,z] = np.sin(psi[:,:,i] - psi[:,:,z])
        
        outputs['psi_ddot'] = np.sum(K_out*S,3)
        outputs['psi_dot'] = dpsi_ds


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

        partials['psi_dot','dpsi_ds'][:] = np.ones((num_nodes*k*tube_nbr)).flatten()
        partials['psi_ddot','psi'] = Pk_ps.flatten()
        partials['psi_ddot','K_out'] = Pp_pk.flatten()

class ODE2system(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=4, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('num_nodes', default=30, types=int)

    def setup(self):
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # input
        self.add_input('R',shape=(num_nodes,k,3,3))
        self.add_input('p',shape=(num_nodes,k,3,1))
        self.add_input('uhat',shape=(num_nodes,k,3,3))
        
        # output
        self.add_output('R_dot',shape=(num_nodes,k,3,3))
        self.add_output('p_dot',shape=(num_nodes,k,3,1))
        
        # partials
        
        row_indices = np.outer(np.arange(num_nodes*k*3*3),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(num_nodes*k*3),np.outer(np.ones(3),np.array([0,1,2])).flatten()) \
                                + (np.arange(0,num_nodes*k*3*3,3).reshape(-1,1))
        self.declare_partials('R_dot', 'R',rows = row_indices, cols = col_indices.flatten())
        row_indices_h = np.outer(np.arange(num_nodes*k*3*3),np.ones(9)).flatten()
        col_indices_h = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2,3,4,5,6,7,8])).flatten()) \
                                + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
        self.declare_partials('R_dot','uhat', rows= row_indices_h.flatten(), cols = col_indices_h.flatten())
        row_indices = np.outer(np.arange(num_nodes*k*3),np.ones(9)).flatten()
        col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2,3,4,5,6,7,8])).flatten()) + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
        self.declare_partials('p_dot', 'R',rows = row_indices, cols = col_indices.flatten())
        self.declare_partials('p_dot', 'p',val=0)
        
    def compute(self,inputs,outputs):
        
        
        R = inputs['R']
        uhat = inputs['uhat']
        e3 = np.zeros((3,1))
        e3[2,:] = 1 
        outputs['p_dot'] = R @ e3
        outputs['R_dot'] = R @ uhat

        
    def compute_partials(self, inputs, partials):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        R = inputs['R']
        uhat = inputs['uhat']
        
        'R'
        Prd_pr = np.zeros((num_nodes,k,9,3))
        

        partials['R_dot','R'][:] = np.tile(np.transpose(uhat,(0,1,3,2)),(3,1)).flatten()

        'uhat'
        Prd_ph = np.zeros((num_nodes,k,9,9))
        Prd_ph[:,:,0,0] = R[:,:,0,0]
        Prd_ph[:,:,0,3] = R[:,:,0,1]
        Prd_ph[:,:,0,6] = R[:,:,0,2]
        Prd_ph[:,:,1,1] = R[:,:,0,0]
        Prd_ph[:,:,1,4] = R[:,:,0,1]
        Prd_ph[:,:,1,7] = R[:,:,0,2]
        Prd_ph[:,:,2,2] = R[:,:,0,0]
        Prd_ph[:,:,2,5] = R[:,:,0,1]
        Prd_ph[:,:,2,8] = R[:,:,0,2]

        Prd_ph[:,:,3,0] = R[:,:,1,0]
        Prd_ph[:,:,3,3] = R[:,:,1,1]
        Prd_ph[:,:,3,6] = R[:,:,1,2]
        Prd_ph[:,:,4,1] = R[:,:,1,0]
        Prd_ph[:,:,4,4] = R[:,:,1,1]
        Prd_ph[:,:,4,7] = R[:,:,1,2]
        Prd_ph[:,:,5,2] = R[:,:,1,0]
        Prd_ph[:,:,5,5] = R[:,:,1,1]
        Prd_ph[:,:,5,8] = R[:,:,1,2]

        Prd_ph[:,:,6,0] = R[:,:,2,0]
        Prd_ph[:,:,6,3] = R[:,:,2,1]
        Prd_ph[:,:,6,6] = R[:,:,2,2]
        Prd_ph[:,:,7,1] = R[:,:,2,0]
        Prd_ph[:,:,7,4] = R[:,:,2,1]
        Prd_ph[:,:,7,7] = R[:,:,2,2]
        Prd_ph[:,:,8,2] = R[:,:,2,0]
        Prd_ph[:,:,8,5] = R[:,:,2,1]
        Prd_ph[:,:,8,8] = R[:,:,2,2]

        'R'
        Ppd_pr = np.zeros((num_nodes,k,3,9))
        Ppd_pr[:,:,0,2] = 1
        Ppd_pr[:,:,1,5] = 1
        Ppd_pr[:,:,2,8] = 1
    
        partials['p_dot','R'][:] = Ppd_pr.flatten()
        partials['R_dot','uhat'][:] = Prd_ph.flatten()


        
class ODEGroup(Group):
    def initialize(self):
        self.options.declare('tube_nbr', default=4, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('num_nodes', default=30, types=int)

    def setup(self):
        n = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        self.add_subsystem('ODE1', ODE1system(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u1_comp', U1Comp(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u2_comp', U2Comp(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u3_comp', U3Comp(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u_comp', UComp(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u_hat', UhatComp(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('initR_comp', InitialRComp(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('ODE2', ODE2system(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])


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
    
    
    comp = ODEGroup(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('Scomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)