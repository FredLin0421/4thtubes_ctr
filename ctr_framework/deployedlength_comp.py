import numpy as np
from openmdao.api import ExplicitComponent


class DeployedlengthComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)
        self.options.declare('tube_nbr', default=3, types=int)

        

    def setup(self):
        #Inputs
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        self.add_input('tube_section_length',shape=(1,tube_nbr))
        self.add_input('beta',shape=(k,tube_nbr))

        self.add_output('deployedlength12constraint',shape=(1,k))
        self.add_output('deployedlength23constraint',shape=(1,k))
        self.add_output('deployedlength34constraint',shape=(1,k))
        self.add_output('deployedlength',shape=(k,tube_nbr))
        
        # partials
        # define indices
        row_indices = np.outer(np.arange(k),np.ones(tube_nbr)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(1),np.arange(tube_nbr)).flatten()) + (np.arange(0,k*tube_nbr,tube_nbr).reshape(-1,1))
        row_indices_t = np.outer(np.arange(k),np.ones(tube_nbr)).flatten()
        col_indices_t = np.outer(np.ones(k),np.arange(tube_nbr)).flatten()
        row_indices_d = np.arange(k*tube_nbr)
        col_indices_d = np.tile(np.arange(tube_nbr), k)
        self.declare_partials('deployedlength12constraint','tube_section_length',rows=row_indices_t.flatten(),cols=col_indices_t.flatten())
        self.declare_partials('deployedlength12constraint','beta',rows=row_indices.flatten(),cols=col_indices.flatten())
        self.declare_partials('deployedlength23constraint','tube_section_length',rows=row_indices_t.flatten(),cols=col_indices_t.flatten())
        self.declare_partials('deployedlength23constraint','beta',rows=row_indices.flatten(),cols=col_indices.flatten())
        self.declare_partials('deployedlength34constraint','tube_section_length',rows=row_indices_t.flatten(),cols=col_indices_t.flatten())
        self.declare_partials('deployedlength34constraint','beta',rows=row_indices.flatten(),cols=col_indices.flatten())
        self.declare_partials('deployedlength','tube_section_length',rows=row_indices_d,cols=col_indices_d)
        self.declare_partials('deployedlength','beta')
        
    def compute(self,inputs,outputs):
        
        
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        tube_section_length = inputs['tube_section_length']
        beta = inputs['beta']
        deployed_length = np.zeros((k,tube_nbr))
        deployed_length = tube_section_length + beta
        constraint12 = np.zeros((1,k))
        constraint23 = np.zeros((1,k))
        constraint34 = np.zeros((1,k))
        constraint12 = deployed_length[:,0] - deployed_length[:,1]
        constraint23 = deployed_length[:,1] - deployed_length[:,2]
        constraint34 = deployed_length[:,2] - deployed_length[:,3]

        outputs['deployedlength12constraint'] = np.reshape(constraint12,(1,k))
        outputs['deployedlength23constraint'] = np.reshape(constraint23,(1,k))
        outputs['deployedlength34constraint'] = np.reshape(constraint34,(1,k))
        outputs['deployedlength'] = deployed_length
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        
        '''Computing Partials'''
        pc12_pb = np.zeros((k,tube_nbr))
        pc12_pb[:,0] = 1
        pc12_pb[:,1] = -1

        pc23_pb = np.zeros((k,tube_nbr))
        pc23_pb[:,1] = 1
        pc23_pb[:,2] = -1

        pc34_pb = np.zeros((k,tube_nbr))
        pc34_pb[:,2] = 1
        pc34_pb[:,3] = -1

        pc12_pt = np.zeros((k,tube_nbr))
        pc12_pt[:,0] = 1
        pc12_pt[:,1] = -1

        pc23_pt = np.zeros((k,tube_nbr))
        pc23_pt[:,1] = 1
        pc23_pt[:,2] = -1

        pc34_pt = np.zeros((k,tube_nbr))
        pc34_pt[:,2] = 1
        pc34_pt[:,3] = -1

        

        partials['deployedlength12constraint','tube_section_length'] = pc12_pt.flatten()
        partials['deployedlength12constraint','beta'][:] = pc12_pb.flatten()
        partials['deployedlength23constraint','tube_section_length'] = pc23_pt.flatten()
        partials['deployedlength23constraint','beta'][:] = pc23_pb.flatten()
        partials['deployedlength34constraint','tube_section_length'] = pc34_pt.flatten()
        partials['deployedlength34constraint','beta'][:] = pc34_pb.flatten()
        partials['deployedlength','tube_section_length'] = 1
        partials['deployedlength','beta'][:] = np.identity(k*tube_nbr)

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

    tube_nbr = 4
    k=7
    comp.add_output('tube_section_length',val = [175,120,65,50])
    beta_init = np.zeros((k,tube_nbr))
    beta_init[:,0] = -55
    beta_init[:,1] = -40
    beta_init[:,2] = -25

    comp.add_output('beta', val=beta_init)
    
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = DeployedlengthComp(k=k,tube_nbr=tube_nbr)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.model.list_outputs()
    
    
