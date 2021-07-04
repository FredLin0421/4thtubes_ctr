import numpy as np
from openmdao.api import ExplicitComponent


class SumkmComp(ExplicitComponent):

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
        self.add_input('tube_ends', shape=(num_nodes,k,tube_nbr))
        self.add_input('kb1')
        self.add_input('kb2')
        self.add_input('kb3')
        self.add_input('kb4')

        # outputs
        self.add_output('sumkm',shape=(num_nodes,k,tube_nbr,tube_nbr))


        # partials

        ''' kb '''
        self.declare_partials('sumkm', 'kb1')
        self.declare_partials('sumkm', 'kb2')
        self.declare_partials('sumkm', 'kb3')
        self.declare_partials('sumkm', 'kb4')
        
        '''tube_ends'''
        
        row_indices = np.outer(np.arange(0,num_nodes*k*tube_nbr**2),np.ones(tube_nbr))
        # col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(tube_nbr**2),np.arange(tube_nbr)).flatten()) + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        self.declare_partials('sumkm', 'tube_ends',rows = row_indices.flatten(),cols = col_indices.flatten())
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kb4 = inputs['kb4']
        tube_ends = inputs['tube_ends']

        tube1 = kb1 * tube_ends[:,:,0]
        tube2 = kb2 * tube_ends[:,:,1]
        tube3 = kb3 * tube_ends[:,:,2]
        tube4 = kb4 * tube_ends[:,:,3]

        sumkm = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        tube_sum = np.zeros((num_nodes,k))
        tube_sum = tube1 + tube2 + tube3 + tube4
        

        sumkm[:,:,:,:] = tube_sum[:,:,np.newaxis,np.newaxis]
        outputs['sumkm'] = sumkm
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        tube_ends = inputs['tube_ends']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kb4 = inputs['kb4']
        Pk_pkb1 = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        Pk_pkb2 = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        Pk_pkb3 = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        Pk_pkb4 = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        for i in range(tube_nbr):
            for j in range(tube_nbr):
                Pk_pkb1[:,:,i,j] = tube_ends[:,:,0]
                Pk_pkb2[:,:,i,j] = tube_ends[:,:,1]
                Pk_pkb3[:,:,i,j] = tube_ends[:,:,2]
                Pk_pkb4[:,:,i,j] = tube_ends[:,:,3]

        partials['sumkm','kb1'][:] =  Pk_pkb1.reshape((num_nodes*k*tube_nbr*tube_nbr,1))
        partials['sumkm','kb2'][:] =  Pk_pkb2.reshape((num_nodes*k*tube_nbr*tube_nbr,1))
        partials['sumkm','kb3'][:] =  Pk_pkb3.reshape((num_nodes*k*tube_nbr*tube_nbr,1))
        partials['sumkm','kb4'][:] =  Pk_pkb4.reshape((num_nodes*k*tube_nbr*tube_nbr,1))
        
        Psk_pt = np.zeros((num_nodes,k,tube_nbr**2,tube_nbr))
        Psk_pt[:,:,:,0] = kb1
        Psk_pt[:,:,:,1] = kb2
        Psk_pt[:,:,:,2] = kb3
        Psk_pt[:,:,:,3] = kb4

        partials['sumkm','tube_ends'][:] = Psk_pt.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 10
    k = 1
    comp = IndepVarComp()
    tube_nbr = 4
    comp.add_output('kb1', val=1.65)
    comp.add_output('kb2', val=5.81)
    comp.add_output('kb3', val=70.35)
    comp.add_output('kb4', val=70.35)
    tube_val = np.zeros((n,k,tube_nbr))
    tube_val[0,:,0] = 1
    tube_val[1:,:,0] = 1
    tube_val[2,:,0] = 1

    tube_val[0,:,1] = 1
    tube_val[1,:,1] = 1
    tube_val[2,:,1] = 0

    tube_val[0,:,2] = 1
    tube_val[1,:,2] = 0
    tube_val[2,:,2] = 0

    tube_val[0,:,3] = 1
    tube_val[1,:,3] = 0
    tube_val[2,:,3] = 0

    comp.add_output('tube_ends', val = tube_val)
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    comp = SumkmComp(k=k,num_nodes=n,tube_nbr=tube_nbr)
    group.add_subsystem('sumkcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
    