import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem

class ODE1profile(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=4, types=int)
        self.options.declare('k', default=54, types=int)
        self.options.declare('num_nodes', default=10, types=int)

    def setup(self):
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # input
        self.add_input('psi',shape=(num_nodes,k,tube_nbr))
        self.add_input('dpsi_ds',shape=(num_nodes,k,tube_nbr))
        
        # output
        self.add_output('psi_',shape=(num_nodes,k,tube_nbr))
        self.add_output('dpsi_ds_',shape=(num_nodes,k,tube_nbr))
        self.add_output('torsionconstraint',shape=(k,tube_nbr))
        
        # partials
        val = np.ones(num_nodes*k*tube_nbr)
        rows = np.arange(num_nodes*k*tube_nbr)
        cols = np.arange(num_nodes*k*tube_nbr)
        self.declare_partials('psi_', 'psi',rows = rows, cols = cols,val=val)
        self.declare_partials('dpsi_ds_', 'dpsi_ds',rows = rows, cols = cols,val=val)
        rows_t = np.arange(k*tube_nbr) 
        cols_t = np.arange((num_nodes-1)*k*tube_nbr,(num_nodes*k*tube_nbr))
        val_t = np.ones(k*tube_nbr)
        self.declare_partials('torsionconstraint','dpsi_ds',rows=rows_t,cols=cols_t,val=val_t)
        
    def compute(self,inputs,outputs):
        
        outputs['psi_'] = inputs['psi']
        outputs['dpsi_ds_'] = inputs['dpsi_ds']
        outputs['torsionconstraint'] = inputs['dpsi_ds'][-1,:,:]

class ODE2profile(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=54, types=int)
        self.options.declare('num_nodes', default=10, types=int)

    def setup(self):
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # input
        self.add_input('R',shape=(num_nodes,k,3,3))
        self.add_input('p',shape=(num_nodes,k,3,1))
        self.add_input('tube_ends_tip',shape=(k,tube_nbr))
        # output
        self.add_output('R_',shape=(num_nodes,k,3,3))
        self.add_output('p_',shape=(num_nodes,k,3,1))
        self.add_output('tip_R',shape=(k,3,3))
        self.add_output('tip_p',shape=(k,3,1))
        # partials
        val = np.ones(num_nodes*k*3*3)
        rows = np.arange(num_nodes*k*3*3)
        cols = np.arange(num_nodes*k*3*3)
        self.declare_partials('R_', 'R',rows = rows, cols = cols,val=val)
        valp = np.ones(num_nodes*k*3)
        rowsp = np.arange(num_nodes*k*3)
        colsp = np.arange(num_nodes*k*3)
        self.declare_partials('p_', 'p',rows = rowsp, cols = colsp,val=valp)

        row_indices = np.outer(np.arange(k*3),np.ones(tube_nbr)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.arange(tube_nbr)).flatten()) + (np.arange(0,k*tube_nbr,tube_nbr).reshape(-1,1))
        self.declare_partials('tip_p', 'p')
        self.declare_partials('tip_R', 'R')
        self.declare_partials('tip_p', 'tube_ends_tip',rows=row_indices, cols=col_indices.flatten())
        
    def compute(self,inputs,outputs):
        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        p = inputs['p']
        R = inputs['R']
        tube_ends_tip = inputs['tube_ends_tip']
        
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

        outputs['tip_p'] = interpolation_val
        outputs['tip_R'] = R[interpolation_idx_r,np.arange(k),:,:]
        outputs['p_'] = inputs['p'] 
        outputs['R_'] = inputs['R']
    def compute_partials(self,inputs,partials):
        
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        p = inputs['p']
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

        partials['tip_p','tube_ends_tip'][:] = pd_pt.flatten()
        partials['tip_p','p'][:]= np.reshape(pd_pp,(k*3,num_nodes*k*3))
        partials['tip_R','R'][:]= po_pt

# class ODEprofiles(Group):
#     def initialize(self):
#         self.options.declare('tube_nbr', default=4, types=int)
#         self.options.declare('k', default=1, types=int)
#         self.options.declare('num_nodes', default=30, types=int)

#     def setup(self):
#         n = self.options['num_nodes']
#         k = self.options['k']
#         tube_nbr = self.options['tube_nbr']

#         self.add_subsystem('ODE1profile', ODE1profile(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])
#         self.add_subsystem('ODE2profile', ODE2profile(num_nodes = n,k=k,tube_nbr=tube_nbr), promotes = ['*'])


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=11
    k=1
    tube_nbr = 4
    comp.add_output('psi', val=np.random.random((n,k,tube_nbr)))
    # comp.add_output('tube_ends_tip', val=np.random.random((k,tube_nbr))*50)
    comp.add_output('p', val=np.random.random((n,k,3,1)))
    comp.add_output('R', val=np.random.random((n,k,3,3)))
    comp.add_output('tube_ends_tip', val=([11,7,4,1]))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    comp = ODE2profile(num_nodes=n,k=k,tube_nbr=tube_nbr)
    group.add_subsystem('bbpoints', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)