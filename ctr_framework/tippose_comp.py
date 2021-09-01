import numpy as np
from openmdao.api import ExplicitComponent


class TipposeComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        #Inputs
        self.add_input('R_', shape=(num_nodes,k,3,3))
        self.add_input('p_', shape=(num_nodes,k,3,1))
        self.add_input('tube_ends_tip',shape=(k,tube_nbr))

        # outputs
        self.add_output('tippos',shape=(k,3))
        self.add_output('tipori',shape=(k,3,3))
        
        row_indices = np.outer(np.arange(k*3),np.ones(tube_nbr)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.arange(tube_nbr)).flatten()) + (np.arange(0,k*tube_nbr,tube_nbr).reshape(-1,1))
        # col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.arange(tube_nbr)).flatten()) + (np.arange(0,k*tube_nbr,tube_nbr).reshape(-1,1))
        self.declare_partials('tippos', 'p_')
        self.declare_partials('tipori', 'R_')
        self.declare_partials('tippos', 'tube_ends_tip',rows=row_indices, cols=col_indices.flatten())

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        p = inputs['p_']
        R = inputs['R_']
        tube_ends_tip = inputs['tube_ends_tip']
        tube_nbr = self.options['tube_nbr']

        p = np.reshape(p,(num_nodes,k,3))
        interpolation_idx_r = np.zeros((k,3))
        interpolation_idx_l = np.zeros((k,3))
        interpolation_val = np.zeros((k,3))
        interpolation_idx_r = np.floor(tube_ends_tip[:,0]).astype(int)
        interpolation_idx_l = np.floor(tube_ends_tip[:,0]).astype(int) - 1
        self.interpolation_idx_r = interpolation_idx_r
        self.interpolation_idx_l = interpolation_idx_l
        tmp = np.ones((k))
        tmp = tube_ends_tip[:,0] - np.floor(tube_ends_tip[:,0])
        self.tmp = tmp
        
        
        interpolation_val[:,0] = p[interpolation_idx_l,np.arange(k),0] \
                +  tmp * (p[interpolation_idx_r,np.arange(k),0] - p[interpolation_idx_l,np.arange(k),0])
        interpolation_val[:,1] = p[interpolation_idx_l,np.arange(k),1] \
                +  tmp * (p[interpolation_idx_r,np.arange(k),1] - p[interpolation_idx_l,np.arange(k),1])
        interpolation_val[:,2] = p[interpolation_idx_l,np.arange(k),2] \
                +  tmp * (p[interpolation_idx_r,np.arange(k),2] - p[interpolation_idx_l,np.arange(k),2])
        

        outputs['tippos'] = interpolation_val
        outputs['tipori'] = R[interpolation_idx_r,np.arange(k),:,:]


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        p = inputs['p_']
        interpolation_idx_r = self.interpolation_idx_r
        interpolation_idx_l = self.interpolation_idx_l
        tmp = self.tmp
        '''Computing Partials'''
        pd_pp = np.zeros((k*3,num_nodes*k*3))
        k_ = np.arange(0,k*3,3)
        pd_pp[np.arange(k)*3,(interpolation_idx_r)*k*3+k_] = tmp
        pd_pp[np.arange(k)*3+1,(interpolation_idx_r)*k*3+k_+1] = tmp
        pd_pp[np.arange(k)*3+2,(interpolation_idx_r)*k*3+k_+2] = tmp
        pd_pp[np.arange(k)*3,(interpolation_idx_l)*k*3+k_] = 1-tmp
        pd_pp[np.arange(k)*3+1,(interpolation_idx_l)*k*3+k_+1] = 1-tmp
        pd_pp[np.arange(k)*3+2,(interpolation_idx_l)*k*3+k_+2] = 1-tmp
        
        pd_pt = np.zeros((k,tube_nbr*3))
        pd_pt[:,0] = (p[interpolation_idx_r,np.arange(k),0] - p[interpolation_idx_l,np.arange(k),0]).squeeze()
        pd_pt[:,4] = (p[interpolation_idx_r,np.arange(k),1] - p[interpolation_idx_l,np.arange(k),1]).squeeze()
        pd_pt[:,8] = (p[interpolation_idx_r,np.arange(k),2] - p[interpolation_idx_l,np.arange(k),2]).squeeze()
        
        po_pt = np.zeros((k*3*3,num_nodes*k*3*3))
        for i in range(9):
            po_pt[np.arange(k)*9+i,(interpolation_idx_r)*k*3*3+k_+i] = 1


        partials['tippos','tube_ends_tip'][:] = pd_pt.flatten()
        partials['tippos','p_'][:]= np.reshape(pd_pp,(k*3,num_nodes*k*3))
        partials['tipori','R_'][:]= po_pt

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=13
    k=1
    comp = IndepVarComp()
    comp.add_output('p', val=np.random.random((n,k,3,1)))
    comp.add_output('R', val=np.random.random((n,k,3,3)))
    comp.add_output('tube_ends_tip', val=([11,7,4,1]))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TipposeComp(num_nodes=n,k=k,tube_nbr=4)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
