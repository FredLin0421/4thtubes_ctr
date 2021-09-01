import numpy as np
from openmdao.api import ExplicitComponent


class KtensorComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs
        self.add_input('psi', shape=(num_nodes,k,tube_nbr))
        self.add_input('dpsi_ds', shape=(num_nodes,k,tube_nbr))
        self.add_input('straight_ends',shape=(num_nodes,k,tube_nbr))

        # outputs
        self.add_output('u1',shape=(num_nodes,k,tube_nbr,3))
        
        row_indices_S = np.outer(np.arange(0,num_nodes*k*tube_nbr*3),np.ones(tube_nbr))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr*3),np.arange(tube_nbr)).flatten())\
                            + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        
        
        self.declare_partials('u1', 'psi', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
        self.declare_partials('u1', 'dpsi_ds', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
        self.declare_partials('u1', 'straight_ends',rows=row_indices_S.flatten(), cols=col_indices_S.flatten())

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tube_nbr= self.options['tube_nbr']
        dpsi_ds = inputs['dpsi_ds']
        psi = inputs['psi']
        straight_ends = inputs['straight_ends']
        

        
        # compute tensor
        R = np.zeros((num_nodes,k,tube_nbr,3,3))
        psi1 = np.zeros((num_nodes,k,tube_nbr))
        psi1[:,:,0] = psi[:,:,0]
        psi1[:,:,1] = psi[:,:,0]
        psi1[:,:,2] = psi[:,:,0]
        psi1[:,:,3] = psi[:,:,0]
        R[:,:,:,0,0] = np.cos(psi-psi1)
        R[:,:,:,0,1] = -np.sin(psi-psi1)
        R[:,:,:,1,0] = np.sin(psi-psi1)
        R[:,:,:,1,1] = np.cos(psi-psi1)
        R[:,:,:,2,2] = np.ones((num_nodes,k,tube_nbr)) 
        import numpy as np
from openmdao.api import ExplicitComponent
from scipy.linalg import block_diag

class U3Comp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=1, types=int)

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs
        self.add_input('kb1')
        self.add_input('kb2')
        self.add_input('kb3')
        self.add_input('kb4')
        self.add_input('kt1')
        self.add_input('kt2')
        self.add_input('kt3')
        self.add_input('kt4')
        self.add_input('tube_ends',shape=(num_nodes,k,tube_nbr))

        # outputs
        self.add_output('u3',shape=(num_nodes,k,3,3))


        ind_skb1 = np.arange(0,9,1)
        indskb = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        row_indices_skb = (np.tile(ind_skb1,num_nodes*k).reshape(num_nodes*k,len(ind_skb1)) +  indskb).flatten()
        self.declare_partials('u3', 'kb1', rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kb2', rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kb3', rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kb4', rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kt1',rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kt2',rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kt3',rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kt4',rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        row_indices_st = np.outer(np.arange(0,num_nodes*k*3*3),np.ones(tube_nbr))
        col_indices_st = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.arange(tube_nbr)).flatten()) + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        
        
        
        self.declare_partials('u3', 'tube_ends', rows=row_indices_st.flatten(), cols=col_indices_st.flatten())
        

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        num_nodes= self.options['num_nodes']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kb4 = inputs['kb4']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']
        kt4 = inputs['kt4']
        tube_ends = inputs['tube_ends']
        # print(tube_ends)
        u2 = np.zeros((num_nodes,k,3,3))
        tube_ends_kb = np.zeros((num_nodes,k,tube_nbr))
        tube_ends_kb[:,:,0] = tube_ends[:,:,0] * kb1
        tube_ends_kb[:,:,1] = tube_ends[:,:,1] * kb2
        tube_ends_kb[:,:,2] = tube_ends[:,:,2] * kb3
        tube_ends_kb[:,:,3] = tube_ends[:,:,3] * kb4
        
        
        K = np.zeros((num_nodes,k,3,3))
        tube_ends_kt = np.zeros((num_nodes,k,tube_nbr))
        tube_ends_kt[:,:,0] = tube_ends[:,:,0] * kt1
        tube_ends_kt[:,:,1] = tube_ends[:,:,1] * kt2
        tube_ends_kt[:,:,2] = tube_ends[:,:,2] * kt3
        tube_ends_kt[:,:,3] = tube_ends[:,:,3] * kt4
        K[:,:,0,0] =  tube_ends_kb[:,:,0]  + tube_ends_kb[:,:,1] + tube_ends_kb[:,:,2] + tube_ends_kb[:,:,3] + 1e-10
        K[:,:,1,1] =  tube_ends_kb[:,:,0] + tube_ends_kb[:,:,1] + tube_ends_kb[:,:,2] + tube_ends_kb[:,:,3] + 1e-10
        K[:,:,2,2] =  tube_ends_kt[:,:,0] + tube_ends_kt[:,:,1] + tube_ends_kt[:,:,2] + tube_ends_kt[:,:,3] + 1e-10

        outputs['u3'] = K 


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_ends = inputs['tube_ends']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kb4 = inputs['kb4']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']
        kt4 = inputs['kt4']

        
        # partial

        'sk/kb1'
        Psk_pkb1 = np.zeros((num_nodes*k,9))
        Psk_pkb1[:, 0] = tube_ends[:,:,0].flatten()
        Psk_pkb1[:, 4] = tube_ends[:,:,0].flatten()
        partials['u3','kb1'][:] = Psk_pkb1.flatten()
        'sk/kb2'
        Psk_pkb2 = np.zeros((num_nodes*k,9))
        Psk_pkb2[:, 0] = tube_ends[:,:,1].flatten()
        Psk_pkb2[:, 4] = tube_ends[:,:,1].flatten()
        partials['u3','kb2'][:] = Psk_pkb2.flatten()
        'sk/kb3'
        Psk_pkb3 = np.zeros((num_nodes*k,9))
        Psk_pkb3[:, 0] = tube_ends[:,:,2].flatten()
        Psk_pkb3[:, 4] = tube_ends[:,:,2].flatten()
        partials['u3','kb3'][:] = Psk_pkb3.flatten()
        'sk/kb4'
        Psk_pkb3 = np.zeros((num_nodes*k,9))
        Psk_pkb3[:, 0] = tube_ends[:,:,3].flatten()
        Psk_pkb3[:, 4] = tube_ends[:,:,3].flatten()
        partials['u3','kb4'][:] = Psk_pkb3.flatten()


        'sk/kt1'
        Psk_pkt1 = np.zeros((num_nodes,k,9))
        Psk_pkt1[:, :,8] = tube_ends[:,:,0]
        partials['u3','kt1'][:] = Psk_pkt1.flatten()
        'sk/kt2'
        Psk_pkt2 = np.zeros((num_nodes,k,9))
        Psk_pkt2[:, :, 8] = tube_ends[:,:,1]
        partials['u3','kt2'][:] = Psk_pkt2.flatten()
        'sk/kt3'
        Psk_pkt3 = np.zeros((num_nodes,k,9))
        Psk_pkt3[:, :,8] = tube_ends[:,:,2]
        partials['u3','kt3'][:] = Psk_pkt3.flatten()
        'sk/kt4'
        Psk_pkt3 = np.zeros((num_nodes,k,9))
        Psk_pkt3[:, :,8] = tube_ends[:,:,3]
        partials['u3','kt4'][:] = Psk_pkt3.flatten()


        'sk/tube_ends'
        Psk_pt = np.zeros((num_nodes,k,9,tube_nbr))
        Psk_pt[:,:, 0,0] = kb1
        Psk_pt[:,:, 4,0] = kb1
        Psk_pt[:,:, 8,0] = kt1

        Psk_pt[:,:, 0,1] = kb2
        Psk_pt[:,:, 4,1] = kb2
        Psk_pt[:,:, 8,1] = kt2

        Psk_pt[:,:, 0,2] = kb3
        Psk_pt[:,:, 4,2] = kb3
        Psk_pt[:,:, 8,2] = kt3

        Psk_pt[:,:, 0,3] = kb4
        Psk_pt[:,:, 4,3] = kb4
        Psk_pt[:,:, 8,3] = kt4

        partials['u3','tube_ends'][:] = Psk_pt.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=20
    k=7
    tube_nbr = 4
    comp = IndepVarComp()
    comp.add_output('kb1', val=1.654)
    comp.add_output('kb2', val=5.8146)
    comp.add_output('kb3', val=70.3552)
    comp.add_output('kb4', val=70.3552)
    comp.add_output('kt1', val=1.2405)
    comp.add_output('kt2', val=4.3609)
    comp.add_output('kt3', val=140.7105)
    comp.add_output('kt4', val=140.7105)
    tube_init = np.zeros((n,k,tube_nbr))
    tube_init[:,:,0] = 1
    tube_init[:15,:,1] = 1
    tube_init[:10,:,2] = 1
    tube_init[:5,:,3] = 1    
    comp.add_output('tube_ends', val=tube_init)
        
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = U3Comp(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    # prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)

        kappa = np.zeros((num_nodes,k,tube_nbr,3,1))
        kappa[:,:,0,0,:] = np.reshape(straight_ends[:,:,0],(num_nodes,k,1))
        kappa[:,:,1,0,:] = np.reshape(straight_ends[:,:,1],(num_nodes,k,1))
        kappa[:,:,2,0,:] = np.reshape(straight_ends[:,:,2],(num_nodes,k,1))
        kappa[:,:,3,0,:] = np.reshape(straight_ends[:,:,3],(num_nodes,k,1))
        
        u = R @ kappa
        dpsi_ds[:,:,0] =  dpsi_ds[:,:,0] - dpsi_ds[:,:,0]
        dpsi_ds[:,:,1] =  dpsi_ds[:,:,1] - dpsi_ds[:,:,0]
        dpsi_ds[:,:,2] =  dpsi_ds[:,:,2] - dpsi_ds[:,:,0]
        dpsi_ds[:,:,3] =  dpsi_ds[:,:,3] - dpsi_ds[:,:,0]

        u[:,:,:,2,:] = u[:,:,:,2,:] - np.reshape(dpsi_ds,(num_nodes,k,tube_nbr,1))
        outputs['u1'] = np.reshape(u,(num_nodes,k,tube_nbr,3))


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        psi = inputs['psi']
        straight_ends = inputs['straight_ends']
        
        psi1 = np.zeros((num_nodes,k,tube_nbr))
        psi1[:,:,0] = psi[:,:,0]
        psi1[:,:,1] = psi[:,:,0]
        psi1[:,:,2] = psi[:,:,0]
        psi1[:,:,3] = psi[:,:,0]
        kappa = np.zeros((num_nodes,k,tube_nbr,3))
        kappa[:,:,0,0] = straight_ends[:,:,0]
        kappa[:,:,1,0] = straight_ends[:,:,1]
        kappa[:,:,2,0] = straight_ends[:,:,2]
        kappa[:,:,3,0] = straight_ends[:,:,3]
        R = np.zeros((num_nodes,k,tube_nbr,3,3))
        R[:,:,:,0,0] = np.cos(psi-psi1)
        R[:,:,:,0,1] = -np.sin(psi-psi1)
        R[:,:,:,1,0] = np.sin(psi-psi1)
        R[:,:,:,1,1] = np.cos(psi-psi1)
        R[:,:,:,2,2] = np.ones((num_nodes,k,tube_nbr))
        'dpsi'
        Pu1_pds = np.zeros((num_nodes,k,tube_nbr*3,tube_nbr))

        Pu1_pds[:,:, 5,1] = -1
        Pu1_pds[:,:, 8,2] = -1
        Pu1_pds[:,:, 11,3] = -1
        partials['u1','dpsi_ds'] = Pu1_pds.flatten()

        'psi'
        Pu1_ppsi = np.zeros((num_nodes,k,tube_nbr*3,tube_nbr))
        Pu1_ppsi[:,:, 3,0] =  np.sin(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        Pu1_ppsi[:,:, 4,0] = -np.cos(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        Pu1_ppsi[:,:, 7,0] = -np.cos(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]
        Pu1_ppsi[:,:, 6,0] =  np.sin(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]
        Pu1_ppsi[:,:, 10,0] = -np.cos(psi[:,:,3]-psi1[:,:,0]) * kappa[:,:,3,0]
        Pu1_ppsi[:,:, 9,0] =  np.sin(psi[:,:,3]-psi1[:,:,0]) * kappa[:,:,3,0]
        
        Pu1_ppsi[:,:, 4,1] = np.cos(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        Pu1_ppsi[:,:, 3,1] = -np.sin(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        
        Pu1_ppsi[:,:, 7,2] = np.cos(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]
        Pu1_ppsi[:,:, 6,2] = -np.sin(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]

        Pu1_ppsi[:,:, 10,3] = np.cos(psi[:,:,3]-psi1[:,:,0]) * kappa[:,:,3,0]
        Pu1_ppsi[:,:, 9,3] =  -np.sin(psi[:,:,3]-psi1[:,:,0]) * kappa[:,:,3,0]
        partials['u1','psi'] = Pu1_ppsi.flatten()

        'straight_ends'
        Pu1_ps = np.zeros((num_nodes,k,tube_nbr*3,tube_nbr))
        Pu1_ps[:,:, 0,0] = 1
        Pu1_ps[:,:, 3,1] = np.cos(psi[:,:,1]-psi1[:,:,0])
        Pu1_ps[:,:, 4,1] = np.sin(psi[:,:,1]-psi1[:,:,0])
        Pu1_ps[:,:, 7,2] = np.sin(psi[:,:,2]-psi1[:,:,0])
        Pu1_ps[:,:, 6,2] = np.cos(psi[:,:,2]-psi1[:,:,0])
        Pu1_ps[:,:, 10,3] = np.sin(psi[:,:,3]-psi1[:,:,0])
        Pu1_ps[:,:, 9,3] = np.cos(psi[:,:,3]-psi1[:,:,0])

        partials['u1','straight_ends'] = Pu1_ps.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=50
    k=1
    tube_nbr = 4
    comp = IndepVarComp()
    comp.add_output('psi', val=np.random.random((n,k,tube_nbr)))
    comp.add_output('dpsi_ds', val=np.random.random((n,k,tube_nbr)))
    comp.add_output('straight_ends', val=np.random.random((n,k,tube_nbr))) 

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = U1Comp(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
