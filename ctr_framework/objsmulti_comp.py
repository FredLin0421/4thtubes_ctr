import numpy as np
from openmdao.api import ExplicitComponent



class ObjsmultiComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('num_anatomy', default=3, types=int)
        
        
        
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        num_anatomy = self.options['num_anatomy']


        #Inputs
        self.add_input('objs_a1')
        self.add_input('objs_a2')
        
        # outputs
        self.add_output('objsmulti')


        # partials
        
        self.declare_partials('objsmulti', 'objs_a1')
        self.declare_partials('objsmulti', 'objs_a2')
        
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        
        objs_a1 = inputs['objs_a1']
        objs_a2 = inputs['objs_a2']
       
        
        
        outputs['objsmulti'] = objs_a1 + objs_a2



    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        
        
        partials['objsmulti','objs_a1'][:] = 1
        partials['objsmulti','objs_a2'][:] = 1


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 100
    k = 3
    gamma1 = 2
    gamma2 = 1
    gamma3 = 4
    gamma4 = 3
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    

    comp.add_output('objs_a1', val = 20)
    # comp.add_output('sumdistance', val = 10.9)
    comp.add_output('objs_a2', val = 8.8)
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ObjsmultiComp(k=k,num_nodes=n)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    