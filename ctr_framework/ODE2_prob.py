from pickle import TRUE
import openmdao.api as om
import numpy as np
from ozone2.api import ODEProblem, Comp
from ctr_framework.ODE_system import ODE2system
from ctr_framework.ODE1_profile import ODE2profile
import matplotlib.pyplot as plt
import time

class ODE2Problem(ODEProblem):
    def setup(self):
        # Inputs
        k=1
        tube_nbr = 4
        self.add_times(step_vector = 'h')
        self.add_parameter('uhat',shape=(self.num_times+1,k,3,3),dynamic=True)
        
        # ODE variables
        self.add_state('R','R_dot', initial_condition_name='initial_condition_R',shape= (k,3,3))
        self.add_state('p','p_dot', initial_condition_name='initial_condition_p',shape= (k,3,1))

        # Output Variables
        # self.add_field_output('field_output2', coefficients_name='coefficients', state_name='R')
        self.add_profile_output('R_', state_name='R',shape=(k,3,3))
        self.add_profile_output('p_', state_name='p',shape=(k,3,1))
        self.add_profile_output('tip_p', state_name='p')#,shape=(k,3,1))
        self.add_profile_output('tip_R', state_name='R',shape=(k,3,3))
        # ODE and Profile Output system
        self.ode_system = Comp(ODE2system)
        self.profile_outputs_system = Comp(ODE2profile)

if __name__ == '__main__':

    # Script to create optimization problem
    num_nodes = 10
    k=54
    h_initial = 0.01
    tube_nbr = 4
    prob = om.Problem()
    comp = om.IndepVarComp()
    # These outputs must match inputs defined in ODEProblemSample
    comp.add_output('coefficients', shape=num_nodes+1)
    # comp.add_output('tube_ends_tip', val=np.ones((k,tube_nbr))*5)
    comp.add_output('uhat', val = np.random.rand(num_nodes+1,k,3,3))
    comp.add_output('initial_condition_R',val = np.random.rand(k,3,3)) # check
    comp.add_output('initial_condition_p',val = np.random.rand(k,3,1))
    comp.add_output('h', shape = num_nodes)
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    ODEProblem_instance = ODE2Problem('RK4', 'time-marching',num_times=num_nodes,display='default', visualization= None)
    # ODEProblem_instance = ODEProblemTest('RK4', 'solver-based',error_tolerance= 0.0000000001,num_times=num,display='default', visualization= 'end')
    comp = ODEProblem_instance.create_component()
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(mode = 'rev')
    coefs = np.zeros(num_nodes+1)
    coefs[num_nodes] = 1.
    # prob.set_val('coefficients', coefs)
    # prob.set_val('R', np.random.random((num_nodes,k,3,3)))
    # prob.set_val('uhat', np.random.random((num_nodes,k,3,3)))
    # prob.set_val('R0', np.random.rand(k,3,3))
    # prob.set_val('h',np.ones(num_nodes)*h_initial)
    t1 = time.time()
    prob.run_model()
    # prob.check_totals(of = ['field_output'], wrt = ['R','uhat','R0','h'])
    # ODEProblem_instance.check_partials(['ODE', 'p_'])
    print('run time: ',time.time() - t1)

    # print(prob.get_val('field_output').shape)
    print(prob.get_val('tip_p'))

    plt.show()
