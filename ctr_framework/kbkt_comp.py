from typing import no_type_check
from numpy.core.numeric import ones
import numpy as np
from openmdao.api import ExplicitComponent

class KbktComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=1, types=int)

        

    def setup(self):
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']

        #Inputs

        self.add_input('kb',shape=(tube_nbr))
        self.add_input('kt',shape=(tube_nbr))
        self.add_input('tube_ends',shape=(num_nodes,k,tube_nbr))

        # outputs
        self.add_output('tubeends_kb',shape=(num_nodes,k,tube_nbr))
        self.add_output('tubeends_kt',shape=(num_nodes,k,tube_nbr))
        self.add_output('ktensor',shape=(num_nodes,k,tube_nbr,3,3))
        self.add_output('u3',shape=(num_nodes,k,3,3))

        
        row_e = np.arange(num_nodes*k*tube_nbr)
        col_k = np.outer(np.ones(num_nodes*k),np.arange(tube_nbr)).flatten()
        self.declare_partials('tubeends_kb', 'kb',rows = row_e, cols = col_k)
        self.declare_partials('tubeends_kt', 'kt',rows = row_e, cols = col_k)
        self.declare_partials('tubeends_kb', 'tube_ends',rows = row_e,cols = row_e)
        self.declare_partials('tubeends_kt', 'tube_ends',rows = row_e,cols = row_e)
        self.declare_partials('tubeends_kt', 'tube_ends')

        col_kb = np.outer(np.ones(num_nodes*k),np.tile(np.arange(tube_nbr),(2,1)).flatten('F')).flatten()
        col_te = np.outer(np.arange(num_nodes*k*tube_nbr),np.ones(3))
        row_te = np.outer(np.ones(num_nodes*k*tube_nbr),np.outer(np.arange(0,3*3,4),np.ones(1)).flatten()) + (np.arange(0,num_nodes*k*tube_nbr*3*3,9).reshape(-1,1))
        row_kb = np.outer(np.ones(num_nodes*k*tube_nbr),np.outer(np.array([0,4]),np.ones(1)).flatten()) + (np.arange(0,num_nodes*k*tube_nbr*3*3,9).reshape(-1,1))
        
        self.declare_partials('ktensor', 'tube_ends',rows=row_te.flatten(),cols=col_te.flatten())
        self.declare_partials('ktensor', 'kb',rows=row_kb.flatten(),cols=col_kb.flatten())
        row_kt = np.ones(num_nodes*k*tube_nbr)*8 + np.arange(0,num_nodes*k*9*tube_nbr,9)
        col_kt = np.tile(np.arange(tube_nbr),num_nodes*k)
        self.declare_partials('ktensor', 'kt',rows = row_kt.flatten(),cols = col_kt)

        col_skb = np.tile(np.arange(tube_nbr),num_nodes*k*2)
        tmpkt = np.ones(num_nodes*k)*8 + np.arange(0,num_nodes*k*9,9)
        col_skt = np.tile(np.arange(tube_nbr),num_nodes*k)
        self.declare_partials('u3', 'kt',rows = np.tile(tmpkt,(tube_nbr,1)).flatten('F'),cols = col_skt)
        col_ste = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.arange(tube_nbr)).flatten()) + (np.arange(0,num_nodes*k*tube_nbr,tube_nbr).reshape(-1,1))
        row_ste = np.outer(np.ones(num_nodes*k),np.outer(np.arange(0,3*3,4),np.ones(tube_nbr)).flatten()) + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
        row_skb = np.outer(np.ones(num_nodes*k),np.outer(np.array([0,4]),np.ones(tube_nbr)).flatten()) + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
        self.declare_partials('u3', 'tube_ends',rows=row_ste.flatten(),cols=col_ste.flatten())
        self.declare_partials('u3', 'kb',rows=row_skb.flatten(),cols=col_skb)



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        num_nodes= self.options['num_nodes']
        kb = inputs['kb']
        kt = inputs['kt']
        tube_ends = inputs['tube_ends']
        
        
        Ktensor = np.zeros((num_nodes,k,tube_nbr,3,3))
        Ktensor[:,:,:,0,0] = tube_ends * kb
        Ktensor[:,:,:,1,1] = tube_ends * kb
        Ktensor[:,:,:,2,2] = tube_ends * kt

        outputs['u3'] = np.sum(Ktensor,2)
        outputs['ktensor'] = Ktensor
        outputs['tubeends_kb'] = tube_ends * kb
        outputs['tubeends_kt'] = tube_ends * kt


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_ends = inputs['tube_ends']
        kb = inputs['kb']
        kt = inputs['kt']
        

        
        # partial
        pt_pt = np.zeros((num_nodes,k,tube_nbr))
        pt_pt[:,:,:] = kb
        partials['tubeends_kb','tube_ends'][:] = pt_pt.flatten()
        pt_pt[:,:,:] = kt
        partials['tubeends_kt','tube_ends'][:] = pt_pt.flatten()

        partials['tubeends_kb','kb'][:] = tube_ends.flatten()
        partials['tubeends_kt','kt'][:] = tube_ends.flatten()

        pk_pkb = np.zeros((num_nodes,k,tube_nbr,2))
        pk_pkb[:,:,:,0] = tube_ends 
        pk_pkb[:,:,:,1] = tube_ends

        psk_pt = np.zeros((num_nodes,k,3,tube_nbr))
        psk_pt[:,:,0,:] = kb
        psk_pt[:,:,1,:] = kb
        psk_pt[:,:,2,:] = kt

        pk_pt = np.zeros((num_nodes,k,tube_nbr,3))
        pk_pt[:,:,:,0] = kb
        pk_pt[:,:,:,1] = kb
        pk_pt[:,:,:,2] = kt

        pk_pkb = np.zeros((num_nodes,k,tube_nbr,2))
        pk_pkb[:,:,:,0] = tube_ends
        pk_pkb[:,:,:,1] = tube_ends

        psk_pkb = np.zeros((num_nodes,k,2,tube_nbr))
        psk_pkb[:,:,0,:] = tube_ends
        psk_pkb[:,:,1,:] = tube_ends
        #pk_pkb[:,:,:,2] = kt

        partials['ktensor','kb'][:] = pk_pkb.flatten()
        partials['ktensor','tube_ends'][:] = pk_pt.flatten()
        partials['ktensor','kt'][:] = tube_ends.flatten()
        
        partials['u3','kb'][:] = psk_pkb.flatten()
        partials['u3','kt'][:] = tube_ends.flatten()
        partials['u3','tube_ends'][:] = psk_pt.flatten()
        

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=10
    k=7
    tube_nbr = 4
    comp = IndepVarComp()
    comp.add_output('kb', val=np.random.rand(tube_nbr))
    comp.add_output('kt', val=np.random.rand(tube_nbr))
    
    tube_init = np.random.rand(n,k,tube_nbr)
    # tube_init[:,:,0] = 1
    # tube_init[:15,:,1] = 1
    # tube_init[:10,:,2] = 1
    # tube_init[:5,:,3] = 1    
    comp.add_output('tube_ends', val=tube_init)
        
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KbktComp(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    # prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    #prob.check_partials(compact_print=False)
