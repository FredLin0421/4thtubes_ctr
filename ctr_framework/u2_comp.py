from re import U
import numpy as np
from openmdao.api import ExplicitComponent
from scipy.linalg import block_diag

class U2Comp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        #Inputs
        
        self.add_input('ktensor',shape=(num_nodes,k,tube_nbr,3,3))
        self.add_input('u1',shape=(num_nodes,k,tube_nbr,3,1))
        
        # outputs
        self.add_output('u2',shape=(num_nodes,k,3,1))
        
        row_indices = np.outer(np.arange(num_nodes*k*3),np.ones(3*tube_nbr))

        col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.arange(3*tube_nbr)).flatten()) \
                            + (np.arange(0,num_nodes*k*tube_nbr*3,tube_nbr*3).reshape(-1,1))
        
        tmp = (np.tile(np.arange(3),(tube_nbr,1))+(np.arange(0,tube_nbr*9,9).reshape(-1,1))).flatten()
        print(np.outer(np.ones(num_nodes*k*3),tmp))
        print(np.tile(np.array([0,3,6]),(num_nodes*k,1)).reshape(-1,1))
        print(np.outer(np.arange(num_nodes)*(tube_nbr*9),np.ones(3)).reshape(-1,1))
        col_indicesk = np.outer(np.ones(num_nodes*k*3),tmp) + np.tile(np.array([0,3,6]),(num_nodes*k,1)).reshape(-1,1) \
                                +np.outer(np.arange(num_nodes*k)*(tube_nbr*9),np.ones(3)).reshape(-1,1)
        print(col_indicesk)
        self.declare_partials('u2','u1',rows=row_indices.flatten(),cols=col_indices.flatten())
        self.declare_partials('u2','ktensor',rows=row_indices.flatten(),cols=col_indicesk.flatten())

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        num_nodes= self.options['num_nodes']
        
        ktensor = inputs['ktensor']
        u1 = inputs['u1']
        outputs['u2'] = np.sum(ktensor @ u1,2)


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        ktensor = inputs['ktensor']
        u1 = inputs['u1'].reshape(num_nodes,k,tube_nbr,3)
        pu_pu1 = np.zeros((num_nodes,k,3,tube_nbr,3))
        pu_pu1[:,:,0,:,0] = ktensor[:,:,:,0,0]
        pu_pu1[:,:,0,:,1] = ktensor[:,:,:,0,1]
        pu_pu1[:,:,0,:,2] = ktensor[:,:,:,0,2]

        pu_pu1[:,:,1,:,0] = ktensor[:,:,:,1,0]
        pu_pu1[:,:,1,:,1] = ktensor[:,:,:,1,1]
        pu_pu1[:,:,1,:,2] = ktensor[:,:,:,1,2]

        pu_pu1[:,:,2,:,0] = ktensor[:,:,:,2,0]
        pu_pu1[:,:,2,:,1] = ktensor[:,:,:,2,1]
        pu_pu1[:,:,2,:,2] = ktensor[:,:,:,2,2]

        pu_pk = np.zeros((num_nodes,k,3,tube_nbr,3))
        pu_pk[:,:,0,:,0] = u1[:,:,:,0]
        pu_pk[:,:,0,:,1] = u1[:,:,:,1]
        pu_pk[:,:,0,:,2] = u1[:,:,:,2]
        pu_pk[:,:,1,:,0] = u1[:,:,:,0]
        pu_pk[:,:,1,:,1] = u1[:,:,:,1]
        pu_pk[:,:,1,:,2] = u1[:,:,:,2]
        pu_pk[:,:,2,:,0] = u1[:,:,:,0]
        pu_pk[:,:,2,:,1] = u1[:,:,:,1]
        pu_pk[:,:,2,:,2] = u1[:,:,:,2]



        partials['u2','u1'] = pu_pu1.flatten()
        partials['u2','ktensor'] = pu_pk.flatten()
        

        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=30
    k=7
    tube_nbr = 4
    comp = IndepVarComp()
    comp.add_output('kb1', val=1.654)
    comp.add_output('kb2', val=5.8146)
    comp.add_output('kb3', val=70.3552)
    comp.add_output('kb4', val=20.3552)
    comp.add_output('ktensor',np.random.rand(n,k,tube_nbr,3,3))
    comp.add_output('u1',np.random.rand(n,k,tube_nbr,3,1))
    
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = U2Comp(num_nodes=n,k=k,tube_nbr = tube_nbr)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    #prob.check_partials(compact_print=False)
