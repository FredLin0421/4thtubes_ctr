import openmdao.api as om
import numpy as np
from ozone2.api import ODEProblem, Comp
from ctr_framework.ODE3_system import ODE3system
from ctr_framework.ODE3_profile import ODE3profile
import matplotlib.pyplot as plt
import time

class ODE3Problem(ODEProblem):
    def setup(self):
        # Inputs
        k=1
        self.add_times(step_vector = 'h')
        self.add_parameter('R_',shape=(self.num_times+1,k,3,3),dynamic=True)
        # ODE variables
        self.add_state('p','dp_ds', initial_condition_name='p0',shape= (k,3,1))

        # Output Variables
        # self.add_field_output('field_output3', coefficients_name='coefficients', state_name='p')
        self.add_profile_output('p_', state_name='p',shape=(k,3,1))
        # ODE and Profile Output system
        self.ode_system = Comp(ODE3system)
        self.profile_outputs_system = Comp(ODE3profile)


if __name__ == '__main__':

    # Script to create optimization problem
    num_nodes = 10
    k=1
    h_initial = 0.01
    prob = om.Problem()
    comp = om.IndepVarComp()
    # These outputs must match inputs defined in ODEProblemSample
    comp.add_output('coefficients', shape=num_nodes+1)
    comp.add_output('R', shape=(num_nodes,k,3,3))
    # comp.add_output('p', shape=(num_nodes,k,3,1))
    comp.add_output('p_0', shape=(k,3,1))
    comp.add_output('h', shape = num_nodes)
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    ODEProblem_instance = ODE3Problem('RK4', 'time-marching',num_times=num_nodes,display='default', visualization= None)
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
    ODEProblem_instance.check_partials(['ODE', 'Profile_output'])
    print('run time: ',time.time() - t1)

    # print(prob.get_val('field_output'))
    print(prob.get_val('p_'))

    plt.show()
