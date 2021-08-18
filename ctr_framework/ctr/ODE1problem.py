import openmdao.api as om
import numpy as np
from ozone2.api import ODEProblem, Comp
from ode1_system import ODE1System
from ode1_profile import ODE1Profile 
import matplotlib.pyplot as plt
import time

# We need to create an ODEProblem class and write the setup method
# def setup is where to declare input and output variables. Similar to Ozone's ODEfunction class.


class ODE1Problem(ODEProblem):

    
    def setup(self):
        
        # Define field outputs, profile outputs, states, parameters, times
        self.add_field_output('field_output', state_name='psi',
                              coefficients_name='coefficients')
        self.add_profile_output('profile_output', state_name='psi')
        self.add_state('psi', 'dpsi_ds', shape=(self.num_times,1,4),initial_condition_name='psi_0')
        self.add_parameter('K_out', shape=(self.num_times,1,4,4), dynamic=True)
        self.add_times(step_vector='h')
        
        # Define ODE and Profile Output systems (OpenMDAO components)
        self.ode_system = Comp(ODE1System)
        self.profile_outputs_system = Comp(ODE1Profile)

        # Define ODE and Profile Output systems (Ozone2 Native components)
        # self.ode_system = ODESystemNative()
        # self.profile_outputs_system = ProfileOutputSystemNative()


# Script to create optimization problem
num_nodes = 5
k = 1
tube_nbr = 4
h_initial = 0.2
prob = om.Problem()
# These outputs must match inputs defined in ODE Problem Sample
comp = om.IndepVarComp()
comp.add_output('coefficients', shape=num_nodes+1)
comp.add_output('K_out', shape=(num_nodes,k,4,4))
comp.add_output('psi_0', shape=(num_nodes,k,tube_nbr))
comp.add_output('h', shape=num_nodes)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

# ODE problem creates OpenMDAO component/group
# ODEProblem_instance = ODEProblemTest('RK4', 'solver-based',num_times=num,display='default', visualization= 'end')
# ODEProblem_instance = ODE1Problem(
#     'ForwardEuler', 'time-marching', num_times=num_nodes, display='default', visualization='None')
# ODEProblem_instance = ODEProblemTest(
#     'ForwardEuler', 'time-marching checkpointing', num_times=num, display='default', visualization='None', num_checkpoints=1)
ODEProblem_instance = ODE1Problem(
    'BackwardEuler', 'time-marching', num_times=num_nodes, display='default', visualization='None')
# ODEProblem_instance = ODEProblemTest(
#     'RK4', 'solver-based', num_times=num, display='default', visualization='None')

comp = ODEProblem_instance.create_component()
prob.model.add_subsystem('comp', comp, promotes=['*'])

prob.setup(mode='rev')
prob.set_val('coefficients', np.ones(num_nodes+1)/(num_nodes+1))
# prob.set_val('param_a', np.array([[-0.5, -0.2]]))
# prob.set_val('y_0', np.array([[1.], [1.]]))
# prob.set_val('y1_0', 1.)
# prob.set_val('h', np.ones(num)*h_initial)

start = time.time()
prob.run_model()
print(time.time() - start)
# ODEProblemInstance.check_partials(['ODE', 'Profile_output'])

# Run and check totals
prob.run_model()
# prob.check_totals(of=['field_output','state2'], wrt=[
#                   'param_a', 'param_b', 'y_0', 'y1_0', 'h'], compact_print=True)
# print(prob.compute_totals())
# print(prob.get_val('total'))
print(prob.get_val('profile_output'))
print(prob.get_val('field_output'))
# print(prob.get_val('field_output2'))
# print(prob.get_val('profile_output2'))

# prob.check_totals(of=['profile_output', 'field_output', 'field_output2', 'profile_output2'], wrt=[
#                   'param_a', 'param_b', 'y_0', 'y1_0', 'h'], compact_print=True)

