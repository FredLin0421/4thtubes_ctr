import numpy as np
from openmdao.api import ExplicitComponent



class ObjtorsionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        
        
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']


        #Inputs
        # self.add_input('dpsi_ds',shape=(num_nodes,k,tube_nbr))
        self.add_input('initial_condision_dpsi',shape=(k,tube_nbr))
        
        # outputs
        self.add_output('objtorsion')


        # partials
        
        self.declare_partials('objtorsion', 'initial_condision_dpsi')

        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        
        initial_condision_dpsi = inputs['initial_condision_dpsi']
       
        
        
        outputs['objtorsion'] = np.linalg.norm(initial_condision_dpsi)



    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        initial_condision_dpsi = inputs['initial_condision_dpsi']
        
        # sumdpsi = sum(sum(sum(initial_condision_dpsi**2))) #+ 1e-10
        sumdpsi = sum(sum(initial_condision_dpsi**2))
        dob_dp = ((sumdpsi)**-0.5) * initial_condision_dpsi
        partials['objtorsion','initial_condision_dpsi'][:] = dob_dp.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 10
    k = 3
    tube_nbr = 4
    gamma1 = 2
    gamma2 = 1
    gamma3 = 4
    gamma4 = 3
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    

    comp.add_output('dpsi_ds', val = np.random.rand(n,k,tube_nbr))
    # comp.add_output('sumdistance', val = 10.9)
    # comp.add_output('objs_a2', val = 8.8)
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ObjtorsionComp(k=k,num_nodes=n,tube_nbr=tube_nbr)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    