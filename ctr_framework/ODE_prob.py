import openmdao.api as om
import numpy as np
from ozone2.api import ODEProblem, Comp
from ctr_framework.ODE_system import ODEGroup
from ctr_framework.ODE1_profile import ODE1profile
import matplotlib.pyplot as plt
import time

class ODEProblem(ODEProblem):
    

    def setup(self):
        
        # Inputs
        k=1
        tube_nbr=4
        self.add_times(step_vector = 'h')
        self.add_parameter('K_out',shape=(self.num_times+1,k,tube_nbr,tube_nbr),dynamic=True)
        # ODE variables
        self.add_state('psi','psi_dot', initial_condition_name='initial_condition_psi',shape= (k,tube_nbr))
        self.add_state('dpsi_ds','psi_ddot', initial_condition_name='initial_condition_dpsi',shape= (k,tube_nbr))
        self.add_state('R','R_dot', initial_condition_name='initial_condition_R',shape= (k,3,3))
        self.add_state('p','p_dot', initial_condition_name='initial_condition_p',shape= (k,3,1))
        # Output Variables
        self.add_profile_output('psi_', state_name='psi',shape=(k,tube_nbr))
        self.add_profile_output('dpsi_ds_', state_name='dpsi_ds',shape=(k,tube_nbr))
        self.add_profile_output('R_', state_name='R',shape=(k,3,3))
        self.add_profile_output('p_', state_name='p',shape=(k,3,1))
        # ODE and Profile Output system
        self.ode_system = Comp(ODEGroup)
        self.profile_outputs_system = Comp(ODE1profile)
# Script to create optimization problem
if __name__ == '__main__':

    num_nodes = 11
    tube_nbr = 4 
    k=1
    h_initial = 0.01
    prob = om.Problem()
    comp = om.IndepVarComp()
    # These outputs must match inputs defined in ODEProblemSample
    comp.add_output('coefficients', shape=num_nodes)
    comp.add_output('K_out', shape=(num_nodes,k,tube_nbr,tube_nbr))
    comp.add_output('uhat', shape=(num_nodes,k,3,3))
    comp.add_output('initial_condition_psi', shape=(k,tube_nbr))
    comp.add_output('initial_condition_dpsi', shape=(k,tube_nbr))
    comp.add_output('initial_condition_R', shape=(k,3,3))
    comp.add_output('initial_condition_p', shape=(k,3,1))
    comp.add_output('h', shape = (num_nodes-1))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    ODEProblem_instance = ODEProblem('RK4', 'time-marching',num_times=num_nodes-1,display='default', visualization= None)
    # ODEProblem_instance = ODEProblemTest('RK4', 'solver-based',error_tolerance= 0.0000000001,num_times=num,display='default', visualization= 'end')
    comp = ODEProblem_instance.create_component()
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    coefs = np.zeros(num_nodes+1)
    coefs[num_nodes] = 1.
    # prob.set_val('coefficients', coefs)
    # prob.set_val('R', np.random.random((num_nodes,k,3,3)))
    # prob.set_val('uhat', np.random.random((num_nodes,k,3,3)))
    # prob.set_val('R0', np.random.rand(k,3,3))
    prob.set_val('h',np.ones(num_nodes-1)*h_initial)
    t1 = time.time()
    prob.run_model()
    # prob.check_totals(of = ['field_output'], wrt = ['R','uhat','R0','h'])
    # ODEProblem_instance.check_partials(['ODE', 'R_'])
    print('run time: ',time.time() - t1)

    # print(prob.get_val('field_output').shape)
    print(prob['psi_'].shape)
    print(prob['p_'].shape)
    # print(prob['torsionconstraint'].shape)

    plt.show()
