import numpy as np
from openmdao.api import ExplicitComponent


class TimevectorComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('tube_nbr', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']

        self.add_input('tube_section_length',shape=(1,tube_nbr))


        self.add_output('h',shape=(num_nodes-1))

        # partials
        
        self.declare_partials('h','tube_section_length')
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        tube_section_length = inputs['tube_section_length']
        outputs['h'] = np.ones(num_nodes-1)*(tube_section_length[:,0] / num_nodes)
        
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        tube_section_length = inputs['tube_section_length']
        
        '''Computing Partials'''

        pf_pt = np.zeros((1,tube_nbr))
        pf_pt[:,0] = 1
        partials['h','tube_section_length'][:] = pf_pt *  (1 / num_nodes) 



if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes = 10
    k = 3
    tube_nbr = 4
    tube = np.random.random((1,tube_nbr))
    

    
    comp.add_output('tube_section_length',val=tube)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = TimevectorComp(tube_nbr = tube_nbr,num_nodes=num_nodes)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.model.list_outputs()
