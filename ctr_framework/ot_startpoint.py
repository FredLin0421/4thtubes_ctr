from openmdao.api import Problem
import openmdao.api as om
import omtools.api as ot
import numpy as np


class otstartpointComp(ot.Group):
    """
    
    """
    def initialize(self):
        self.options.declare('num_cp', default=3, types=int)


    def setup(self):
        # Create independent variable
        num_cp = self.options['num_cp']
        # x1 = self.create_indep_var('x1', val=40)
        cp = self.declare_input('cp',shape=(num_cp,3))
        print(cp.shape)
        print(cp[0,:].shape)
        # startpoint_constraint = cp[0,:]
        startpoint_constraints = self.create_output('startpoint_constraint', shape=(1,3))
        startpoint_constraints[0,:] = cp[0,:]
        # Simple addition        
        self.register_output('startpoint_', startpoint_constraints)

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=50
    k=2
    comp = IndepVarComp()
    comp.add_output('cp', val=np.random.random((n,3)))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])

    # omgroup = ot.Group()

    comp = otstartpointComp(num_cp=n)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])    
    prob = Problem()
    prob.model = group
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.list_outputs()
    
    # print('startpoint_constraint', prob['startpoint_constraint'].shape)
    # print(prob['startpoint_constraint'])