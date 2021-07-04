import numpy as np
from openmdao.api import ExplicitComponent


class TubeclearanceComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=1, types=int)
        self.options.declare('tube_nbr', default=3, types=int)

        

    def setup(self):
        #Inputs

        tube_nbr = self.options['tube_nbr']
        self.add_input('d2')
        self.add_input('d3')
        self.add_input('d4')
        self.add_input('d5')
        self.add_input('d6')
        self.add_input('d7')

        self.add_output('tubeclearanceconstraint',shape=(1,tube_nbr-1))
        

        # partials 
        self.declare_partials('tubeclearanceconstraint','d2')
        self.declare_partials('tubeclearanceconstraint','d3')
        self.declare_partials('tubeclearanceconstraint','d4')
        self.declare_partials('tubeclearanceconstraint','d5')
        self.declare_partials('tubeclearanceconstraint','d6')
        self.declare_partials('tubeclearanceconstraint','d7')
        
    def compute(self,inputs,outputs):
        
        tube_nbr = self.options['tube_nbr']
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        d6 = inputs['d6']
        d7 = inputs['d7']
        constraint = np.zeros((1,tube_nbr-1))
        constraint[:,0] = d3-d2
        constraint[:,1] = d5-d4
        constraint[:,2] = d7-d6
        

        outputs['tubeclearanceconstraint'] = constraint
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        tube_nbr = self.options['tube_nbr']
        '''Computing Partials'''
        pdc_pd2 = np.zeros((tube_nbr-1,1))
        pdc_pd2[0,:] = -1
        pdc_pd3 = np.zeros((tube_nbr-1,1))
        pdc_pd3[0,:] = 1
        pdc_pd4 = np.zeros((tube_nbr-1,1))
        pdc_pd4[1,:] = -1
        pdc_pd5 = np.zeros((tube_nbr-1,1))
        pdc_pd5[1,:] = 1
        pdc_pd6 = np.zeros((tube_nbr-1,1))
        pdc_pd6[2,:] = -1
        pdc_pd7 = np.zeros((tube_nbr-1,1))
        pdc_pd7[2,:] = 1
        
        partials['tubeclearanceconstraint','d2'] = pdc_pd2
        partials['tubeclearanceconstraint','d3'] = pdc_pd3 
        partials['tubeclearanceconstraint','d4'] = pdc_pd4
        partials['tubeclearanceconstraint','d5'] = pdc_pd5
        partials['tubeclearanceconstraint','d6'] = pdc_pd6
        partials['tubeclearanceconstraint','d7'] = pdc_pd7



        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    
    comp.add_output('d2',val=2.5)
    comp.add_output('d3',val=3)
    comp.add_output('d4',val=5)
    comp.add_output('d5',val=6)
    comp.add_output('d6',val=5)
    comp.add_output('d7',val=6)

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = TubeclearanceComp(tube_nbr=4)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    
    
    
