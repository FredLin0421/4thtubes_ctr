import numpy as np
from openmdao.api import ExplicitComponent
import math


class TubeendsComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('tube_nbr', default=4, types=int)
        self.options.declare('a', default = 30, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs
        self.add_input('tube_section_length',shape=(1,tube_nbr))
        self.add_input('beta',shape=(k,tube_nbr))


        # outputs
        self.add_output('tube_ends_hyperbolic',shape=(num_nodes,k,tube_nbr))
        self.add_output('tube_ends_tip',shape=(k,tube_nbr))
        self.add_output('deploy_length',shape=(k,tube_nbr))

        


        # partials
        
        self.declare_partials('tube_ends_hyperbolic','tube_section_length')
        col_indices_b = np.outer(np.ones(num_nodes),np.arange(tube_nbr*k)).flatten()
        row_indices_b = np.arange(num_nodes*k*tube_nbr).flatten()
        self.declare_partials('tube_ends_hyperbolic', 'beta', rows= row_indices_b,cols= col_indices_b)

        row_indices = np.outer(np.arange(0,k*tube_nbr),np.ones(tube_nbr)).flatten()
        # col_indices = np.outer(np.ones(k),np.outer(np.ones(tube_nbr),np.array([0,1,2])).flatten()) + (np.arange(0,k*tube_nbr,tube_nbr).reshape(-1,1))
        col_indices = np.outer(np.ones(k),np.outer(np.ones(tube_nbr),np.arange(tube_nbr)).flatten()) + (np.arange(0,k*tube_nbr,tube_nbr).reshape(-1,1))
        self.declare_partials('tube_ends_tip','tube_section_length')
        self.declare_partials('tube_ends_tip', 'beta' , rows=row_indices,cols=col_indices.flatten())

        row_indices_d = np.arange(k*tube_nbr)
        # col_indices_d = np.outer(np.ones(k),np.array([0,1,2])).flatten()
        col_indices_d = np.outer(np.ones(k),np.arange(tube_nbr)).flatten()
        self.declare_partials('deploy_length','tube_section_length',rows=row_indices_d,cols=col_indices_d)
        self.declare_partials('deploy_length', 'beta')
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        a = self.options['a']
        tube_nbr = self.options['tube_nbr']
        tube_section_length = inputs['tube_section_length']
        beta = inputs['beta']

        # compute the deployed length and apply zero stiffness
        deployed_length = np.zeros((k,tube_nbr))
        deployed_length = tube_section_length + beta
        link_length = tube_section_length[:,0] / num_nodes
        tube_ends = (deployed_length / link_length) 
       
        
        temp = np.zeros((num_nodes,k,tube_nbr))
        
        temp[:,:,0] = 1-(np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends[:,0]))/2 + 0.5)
        temp[:,:,1] = 1-(np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends[:,1]))/2 + 0.5)
        temp[:,:,2] = 1-(np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends[:,2]))/2 + 0.5)
        temp[:,:,3] = 1-(np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends[:,3]))/2 + 0.5)
        
        outputs['tube_ends_hyperbolic'] = temp
        outputs['tube_ends_tip'] = tube_ends
        outputs['deploy_length'] = deployed_length
        


    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        a = self.options['a']
        tube_nbr = self.options['tube_nbr']
        tube_section_length = inputs['tube_section_length']
        beta = inputs['beta']
        deployed_length = tube_section_length + beta
        link_length = tube_section_length[:,0] / num_nodes
        tube_ends = (deployed_length / link_length) 
        
        Pe_pt = np.zeros((k,tube_nbr,tube_nbr))
        Pt_pt = np.zeros((num_nodes,k,tube_nbr,tube_nbr))
        x = (np.outer(np.arange(1,num_nodes+1),np.ones(k)))

        Pe_pt[:, 0,0] = num_nodes/tube_section_length[:,0] - num_nodes*(beta[:,0] + tube_section_length[:,0])/tube_section_length[:,0]**2
        Pt_pt[:,:,0,0] = -0.5*a*(1 - np.tanh(a*(-num_nodes*(beta[:,0] + tube_section_length[:,0])\
                                    /tube_section_length[:,0] + x))**2)*(-num_nodes/tube_section_length[:,0] + num_nodes*(beta[:,0] + tube_section_length[:,0])\
                                    /tube_section_length[:,0]**2)
        for i in range(1,tube_nbr):
            Pe_pt[:, i,0] = -num_nodes*(beta[:,i] + tube_section_length[:,i])/tube_section_length[:,0]**2
            Pe_pt[:, i,i] = (num_nodes/(tube_section_length[:,0]))
            Pt_pt[:,:,i,i] = 0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,i] + tube_section_length[:,i])\
                                    /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
            Pt_pt[:,:,i,0] = -0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,i] + tube_section_length[:,i])\
                                    /tube_section_length[:,0] + x))**2)*(beta[:,i] + tube_section_length[:,i])/tube_section_length[:,0]**2
        
    #  beta
        
        Pe_pb = np.zeros((k,tube_nbr,tube_nbr))
        Pt_pb = np.zeros((num_nodes,k,tube_nbr))
        for i in range(tube_nbr):
            Pe_pb[:, i,i] = (num_nodes/(tube_section_length[:,0]))
            Pt_pb[:,:,i] = 0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,i] + tube_section_length[:,i])\
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        

        partials['tube_ends_tip','tube_section_length'][:] = Pe_pt.reshape((k*tube_nbr,tube_nbr))
        partials['tube_ends_tip','beta'][:] = Pe_pb.flatten()
        partials['tube_ends_hyperbolic','tube_section_length'][:] = Pt_pt.reshape((num_nodes*k*tube_nbr,tube_nbr))
        partials['tube_ends_hyperbolic','beta'][:] = Pt_pb.flatten()
        partials['deploy_length','beta'][:] = np.identity(k*tube_nbr)
        partials['deploy_length','tube_section_length'][:] = np.ones(k*tube_nbr)
        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n = 30
    k = 3
    comp.add_output('tube_section_length',val = [175,120,65,30])
    beta_init = np.zeros((k,4))
    beta_init[:,0] = -20.5
    beta_init[:,1] = -40.7
    beta_init[:,2] = -25.9
    beta_init[:,2] = -25

    comp.add_output('beta', val=beta_init)
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TubeendsComp(num_nodes=n,k=k,tube_nbr=4)
    group.add_subsystem('TubeendsComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    