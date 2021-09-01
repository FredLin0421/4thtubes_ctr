from networkx.algorithms.structuralholes import constraint
import numpy as np
from openmdao.api import ExplicitComponent


class DiameterComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)
        self.options.declare('tube_nbr', default=3, types=int)

        

    def setup(self):
        #Inputs
        tube_nbr = self.options['tube_nbr']
        self.add_input('d1')
        self.add_input('d2')
        self.add_input('d3')
        self.add_input('d4')
        self.add_input('d5')
        self.add_input('d6')
        self.add_input('d7')
        self.add_input('d8')
        #self.add_input('d',shape=(tube_nbr*2))

        self.add_output('diameterconstraint',shape=(tube_nbr))

        # partials        
        self.declare_partials('diameterconstraint','d1')
        self.declare_partials('diameterconstraint','d2')
        self.declare_partials('diameterconstraint','d3')
        self.declare_partials('diameterconstraint','d4')
        self.declare_partials('diameterconstraint','d5')
        self.declare_partials('diameterconstraint','d6')
        self.declare_partials('diameterconstraint','d7')
        self.declare_partials('diameterconstraint','d8')
        '''col = np.arange(tube_nbr*2)
        row = np.outer(np.arange(tube_nbr),np.ones(2))
        
        self.declare_partials('diameterconstraint','d',rows=row.flatten(),cols=col.flatten())'''
        
    def compute(self,inputs,outputs):
        tube_nbr = self.options['tube_nbr']
        d1 = inputs['d1']
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        d6 = inputs['d6']
        d7 = inputs['d7']
        d8 = inputs['d8']
        # d = inputs['d']
        constraint = np.zeros((tube_nbr))
        constraint[0] = d2-d1
        constraint[1] = d4-d3
        constraint[2] = d6-d5
        constraint[3] = d8-d7
        outputs['diameterconstraint'] = constraint
        #outputs['diameterconstraint'] = d[1::2] - d[::2]
        
        

    def compute_partials(self,inputs,partials):
        
        
        
        tube_nbr = self.options['tube_nbr']
        '''Computing Partials'''
        pdc_pd1 = np.zeros((tube_nbr,1))
        pdc_pd1[0,:] = -1
        pdc_pd2 = np.zeros((tube_nbr,1))
        pdc_pd2[0,:] = 1
        pdc_pd3 = np.zeros((tube_nbr,1))
        pdc_pd3[1,:] = -1
        pdc_pd4 = np.zeros((tube_nbr,1))
        pdc_pd4[1,:] = 1
        pdc_pd5 = np.zeros((tube_nbr,1))
        pdc_pd5[2,:] = -1
        pdc_pd6 = np.zeros((tube_nbr,1))
        pdc_pd6[2,:] = 1
        pdc_pd7 = np.zeros((tube_nbr,1))
        pdc_pd7[3,:] = -1
        pdc_pd8 = np.zeros((tube_nbr,1))
        pdc_pd8[3,:] = 1

        
        partials['diameterconstraint','d1'] = pdc_pd1
        partials['diameterconstraint','d2'] = pdc_pd2
        partials['diameterconstraint','d3'] = pdc_pd3 
        partials['diameterconstraint','d4'] = pdc_pd4
        partials['diameterconstraint','d5'] = pdc_pd5
        partials['diameterconstraint','d6'] = pdc_pd6
        partials['diameterconstraint','d7'] = pdc_pd7
        partials['diameterconstraint','d8'] = pdc_pd8
        '''pd_pd = np.zeros((tube_nbr*2))
        pd_pd[1::2] = 1
        pd_pd[::2] = -1
        partials['diameterconstraint','d'] = pd_pd.flatten()'''


        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    tube_nbr = 4
    comp.add_output('d1',val=1)
    comp.add_output('d2',val=2.5)
    comp.add_output('d3',val=3)
    comp.add_output('d4',val=4.5)
    comp.add_output('d5',val=5)
    comp.add_output('d6',val=6)
    comp.add_output('d',val=np.random.rand(1,tube_nbr*2))

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = DiameterComp(tube_nbr = 4)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    
    
