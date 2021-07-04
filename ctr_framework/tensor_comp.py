import numpy as np
from openmdao.api import ExplicitComponent


class TensorComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=40, types=int)
        
        

    '''This class is defining K tensor'''
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

        # outputs
        self.add_output('K',shape=(num_nodes,k,tube_nbr,tube_nbr))


        # partials
        ''' kb '''
        # ind_kb1 = np.arange(0,9,1)
        # indkb = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        ind_kb1 = np.arange(0,tube_nbr**2,1)
        indkb = np.arange(0,num_nodes*k*tube_nbr**2,tube_nbr**2).reshape(-1,1)
        row_indices_kb = (np.tile(ind_kb1,num_nodes*k).reshape(num_nodes*k,len(ind_kb1)) +  indkb).flatten()
        self.declare_partials('K', 'kb1', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('K', 'kb2', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('K', 'kb3', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('K', 'kb4', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())

        ''' kt '''
        ind_kt1 = np.arange(tube_nbr)
        # ind_kt1 = np.array([0,1,2])
        ind_kt2 = np.arange(tube_nbr) + tube_nbr
        ind_kt3 = np.arange(tube_nbr) + tube_nbr*2
        ind_kt4 = np.arange(tube_nbr) + tube_nbr*3
        
        
        # ind_kt3 = np.array([6,7,8])
        indkt = np.arange(0,num_nodes*k*tube_nbr**2,tube_nbr**2).reshape(-1,1)
        row_indices_kt1 = (np.tile(ind_kt1,num_nodes*k).reshape(num_nodes*k,len(ind_kt1)) +  indkt).flatten()
        row_indices_kt2 = (np.tile(ind_kt2,num_nodes*k).reshape(num_nodes*k,len(ind_kt2)) +  indkt).flatten()
        row_indices_kt3 = (np.tile(ind_kt3,num_nodes*k).reshape(num_nodes*k,len(ind_kt3)) +  indkt).flatten()
        row_indices_kt4 = (np.tile(ind_kt4,num_nodes*k).reshape(num_nodes*k,len(ind_kt4)) +  indkt).flatten()

        self.declare_partials('K', 'kt1', rows=row_indices_kt1, cols=np.zeros(len(row_indices_kt1)).flatten())
        self.declare_partials('K', 'kt2', rows=row_indices_kt2, cols=np.zeros(len(row_indices_kt2)).flatten())
        self.declare_partials('K', 'kt3', rows=row_indices_kt3, cols=np.zeros(len(row_indices_kt3)).flatten())
        self.declare_partials('K', 'kt4', rows=row_indices_kt4, cols=np.zeros(len(row_indices_kt4)).flatten())
        

        # '''kappa'''
        # ind_kp1 = np.array([0,1,2,3,6])
        # ind_kp2 = np.array([1,3,4,5,7])
        # ind_kp3 = np.array([2,5,6,7,8])
        # indkp = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
       
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kb4 = inputs['kb4']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']
        kt4 = inputs['kt4']
        
        # compute tensor
        K = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
       
        T_kb = np.ones((num_nodes,k,tube_nbr))
        T_kb[:,:,0] = T_kb[:,:,0] * kb1
        T_kb[:,:,1] = T_kb[:,:,1] * kb2
        T_kb[:,:,2] = T_kb[:,:,2] * kb3
        T_kb[:,:,3] = T_kb[:,:,3] * kb4
        K[:,:,0,:] = T_kb * kb1/kt1  
        K[:,:,1,:] = T_kb * kb2/kt2
        K[:,:,2,:] = T_kb * kb3/kt3
        K[:,:,3,:] = T_kb * kb4/kt4
        
        outputs['K'] = K


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        # kappa = inputs['kappa']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kb4 = inputs['kb4']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']
        kt4 = inputs['kt4']

        Pk_pkb1 = np.zeros((num_nodes*k,tube_nbr**2))
        Pk_pkb1[:, 0] = 2*kb1/kt1
        Pk_pkb1[:, 1] = kb2/kt1
        Pk_pkb1[:, 2] = kb3/kt1
        Pk_pkb1[:, 3] = kb4/kt1
        Pk_pkb1[:, 4] = kb2/kt2
        Pk_pkb1[:, 8] = kb3/kt3
        Pk_pkb1[:, 12] = kb4/kt4
        

        Pk_pkb2 = np.zeros((num_nodes*k,tube_nbr**2))
        Pk_pkb2[:, 1] = kb1/kt1
        Pk_pkb2[:, 4] = kb1/kt2
        Pk_pkb2[:, 5] = 2*kb2/kt2
        Pk_pkb2[:, 6] = kb3/kt2
        Pk_pkb2[:, 7] = kb4/kt2
        Pk_pkb2[:, 9] = kb3/kt3
        Pk_pkb2[:, 13] = kb4/kt4

        Pk_pkb3 = np.zeros((num_nodes*k,tube_nbr**2))
        Pk_pkb3[:, 2] = kb1/kt1
        Pk_pkb3[:, 6] = kb2/kt2
        Pk_pkb3[:, 8] = kb1/kt3
        Pk_pkb3[:, 9] = kb2/kt3
        Pk_pkb3[:, 10] = 2*kb3/kt3
        Pk_pkb3[:, 11] = kb4/kt3
        Pk_pkb3[:, 14] = kb4/kt4

        Pk_pkb4 = np.zeros((num_nodes*k,tube_nbr**2))
        Pk_pkb4[:, 3] = kb1/kt1
        Pk_pkb4[:, 7] = kb2/kt2
        Pk_pkb4[:, 11] = kb3/kt3
        Pk_pkb4[:, 12] = kb1/kt4
        Pk_pkb4[:, 13] = kb2/kt4
        Pk_pkb4[:, 14] = kb3/kt4
        Pk_pkb4[:, 15] = 2*kb4/kt4
        
        ''' kt '''
        Pk_pkt1 = np.zeros((num_nodes*k,tube_nbr))
        Pk_pkt1[:, 0] = -kb1**2/kt1**2
        Pk_pkt1[:, 1] = -kb1*kb2/kt1**2
        Pk_pkt1[:, 2] = -kb1*kb3/kt1**2
        Pk_pkt1[:, 3] = -kb1*kb4/kt1**2

        Pk_pkt2 = np.zeros((num_nodes*k,tube_nbr))
        Pk_pkt2[:, 0] = -kb1*kb2/kt2**2
        Pk_pkt2[:, 1] = -kb2**2/kt2**2
        Pk_pkt2[:, 2] = -kb2*kb3/kt2**2
        Pk_pkt2[:, 3] = -kb2*kb4/kt2**2
      
        Pk_pkt3 = np.zeros((num_nodes*k,tube_nbr))
        Pk_pkt3[:, 0] = -kb1*kb3/kt3**2
        Pk_pkt3[:, 1] = -kb2*kb3/kt3**2
        Pk_pkt3[:, 2] = -kb3**2/kt3**2
        Pk_pkt3[:, 3] = -kb4*kb3/kt3**2

        Pk_pkt4 = np.zeros((num_nodes*k,tube_nbr))
        Pk_pkt4[:, 0] = -kb1*kb4/kt4**2
        Pk_pkt4[:, 1] = -kb2*kb4/kt4**2
        Pk_pkt4[:, 2] = -kb3*kb4/kt4**2
        Pk_pkt4[:, 3] = -kb4**2/kt4**2
 
        
        partials['K','kb1'][:] =  Pk_pkb1.flatten()
        partials['K','kb2'][:] =  Pk_pkb2.flatten()
        partials['K','kb3'][:] =  Pk_pkb3.flatten()
        partials['K','kb4'][:] =  Pk_pkb4.flatten()
        partials['K','kt1'][:] =  Pk_pkt1.flatten()
        partials['K','kt2'][:] =  Pk_pkt2.flatten()
        partials['K','kt3'][:] =  Pk_pkt3.flatten()
        partials['K','kt4'][:] =  Pk_pkt4.flatten()
        



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    
    comp.add_output('kb1', val=0.1)
    comp.add_output('kb2', val=10)
    comp.add_output('kb3', val=2)
    comp.add_output('kb4', val=4)
    comp.add_output('kt1', val=1)
    comp.add_output('kt2', val=4)
    comp.add_output('kt3', val=1)
    comp.add_output('kt4', val=2)
   
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TensorComp(tube_nbr=4)
    group.add_subsystem('tensorcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)