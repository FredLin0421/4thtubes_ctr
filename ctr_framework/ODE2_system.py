import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem
# from ozone2.api import NativeSystem


class ODE2system(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
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
        
        partials['R_dot','R'][:] = np.tile(np.transpose(uhat,(0,1,3,2)),(3,1)).flatten()

        'uhat'
        Prd_ph = np.zeros((num_nodes,k,9,9))
        
        for i in range(3):
            Prd_ph[:,:,3*i,0] = R[:,:,i,i]

        
        Prd_ph[:,:,3*i,0] = R[:,:,0,0]
        Prd_ph[:,:,[0,3,6],3] = R[:,:,0,1]
        Prd_ph[:,:,[0,3,6],6] = R[:,:,0,2]

        Prd_ph[:,:,[0,3,6],0] = R[:,:,0,0]
        Prd_ph[:,:,[0,3,6],3] = R[:,:,0,1]
        Prd_ph[:,:,[0,3,6],6] = R[:,:,0,2]
        
        Prd_ph[:,:,[0,3,6],0] = R[:,:,0,0]
        Prd_ph[:,:,[0,3,6],3] = R[:,:,0,1]
        Prd_ph[:,:,[0,3,6],6] = R[:,:,0,2]
        
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

# class ODE3system(ExplicitComponent):
    
#     def initialize(self):
#         self.options.declare('tube_nbr', default=3, types=int)
#         self.options.declare('k', default=1, types=int)
#         self.options.declare('num_nodes', default=10, types=int)

#     def setup(self):
        
#         tube_nbr = self.options['tube_nbr']
#         num_nodes = self.options['num_nodes']
#         k = self.options['k']
        
#         # input
#         self.add_input('R',shape=(num_nodes,k,3,3))
#         self.add_input('p',shape=(num_nodes,k,3,1))
        
#         # output
#         self.add_output('p_dot',shape=(num_nodes,k,3,1))
        
#         # partials
#         row_indices = np.outer(np.arange(num_nodes*k*3),np.ones(9)).flatten()
#         col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2,3,4,5,6,7,8])).flatten()) + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
#         self.declare_partials('p_dot', 'R',rows = row_indices, cols = col_indices.flatten())
#         self.declare_partials('p_dot', 'p',val=0)
#     def compute(self,inputs,outputs):
        
        
#         R = inputs['R']
#         e3 = np.zeros((3,1))
#         e3[2,:] = 1 
#         outputs['p_dot'] = R @ e3


  
        
#     def compute_partials(self, inputs, partials):
        
#         num_nodes = self.options['num_nodes']
#         k = self.options['k']
        
        
#         'R'
#         Ppd_pr = np.zeros((num_nodes,k,3,9))
#         Ppd_pr[:,:,0,2] = 1
#         Ppd_pr[:,:,1,5] = 1
#         Ppd_pr[:,:,2,8] = 1
    
#         partials['p_dot','R'][:] = Ppd_pr.flatten()

# class ODEGroup(Group):
#     def initialize(self):
#         self.options.declare('num_nodes')

#     def setup(self):
#         n = self.options['num_nodes']
#         self.add_subsystem('comp1', ODE2system(num_nodes = n), promotes = ['*'])
#         self.add_subsystem('comp2', ODE3system(num_nodes = n), promotes = ['*'])

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=175
    k=2

    comp.add_output('R', val=np.random.random((n,k,3,3)))
    comp.add_output('p', val=np.random.random((n,k,3,1)))
    comp.add_output('uhat', val=np.random.random((n,k,3,3)))
    
    
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ODE2system(num_nodes=n,k=k)
    group.add_subsystem('bborientation', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)