import numpy as np
from openmdao.api import ExplicitComponent


class TipvecComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']

        #Inputs
        self.add_input('rot_p', shape=(num_nodes,k,3,1))
        self.add_input('tube_ends_tip',shape=(k,tube_nbr))

        # outputs
        self.add_output('tipvec',shape=(k,3))
        self.add_output('norm',shape=(k,1))
        
        row_indices = np.outer(np.arange(k*3),np.ones(tube_nbr)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.arange(tube_nbr)).flatten()) + (np.arange(0,k*tube_nbr,tube_nbr).reshape(-1,1))
        self.declare_partials('tipvec', 'rot_p')
        self.declare_partials('norm', 'rot_p')

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        rot_p = inputs['rot_p']
        tube_ends_tip = inputs['tube_ends_tip']
        tube_nbr = self.options['tube_nbr']

        rot_p = np.reshape(rot_p,(num_nodes,k,3))
        interpolation_idx_r = np.zeros((k,3))
        interpolation_idx_l = np.zeros((k,3))
        interpolation_val = np.zeros((k,3))
        interpolation_idx_r = np.floor(tube_ends_tip[:,0]).astype(int)
        interpolation_idx_l = np.floor(tube_ends_tip[:,0]).astype(int) - 1
        self.interpolation_idx_r = interpolation_idx_r
        self.interpolation_idx_l = interpolation_idx_l


        
        vec = np.zeros((k,3))
        vec[:,:] = rot_p[interpolation_idx_r,np.arange(k),:] - rot_p[interpolation_idx_l,np.arange(k),:]
        self.vec = vec
        norm = np.linalg.norm(vec,axis=1) 
        interpolation_val[:,:] = vec 
        
        
        outputs['norm'] = norm.reshape(-1,1)
        outputs['tipvec'] = interpolation_val


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        rot_p = inputs['rot_p']
        interpolation_idx_r = self.interpolation_idx_r
        interpolation_idx_l = self.interpolation_idx_l
        vec = self.vec
        '''Computing Partials'''
        pd_pp = np.zeros((k*3,num_nodes*k*3))
        k_ = np.arange(0,k*3,3)
        sumdpsi = np.sum((vec)**2,1)
        pn_pt = np.zeros((k,num_nodes*k*3))
        idx = np.arange(3)
        
        for i in range(3):
            pd_pp[np.arange(k)*3+i,(interpolation_idx_r)*k*3+k_+i] = 1
            pd_pp[np.arange(k)*3+i,(interpolation_idx_l)*k*3+k_+i] = -1
            pn_pt[np.arange(k), (interpolation_idx_r)*k*3 + k_ +i] = (((sumdpsi)**-0.5) * vec[:,i])
            pn_pt[np.arange(k), (interpolation_idx_l)*k*3 + k_ +i] = -(((sumdpsi)**-0.5) * vec[:,i])

        partials['norm','rot_p'][:]= pn_pt
        partials['tipvec','rot_p'][:]= np.reshape(pd_pp,(k*3,num_nodes*k*3))

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=50
    k=1
    comp = IndepVarComp()
    comp.add_output('rot_p', val=np.random.random((n,k,3,1)))
    comp.add_output('tube_ends_tip', val=(40.5,35.6,30,15.1))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TipvecComp(num_nodes=n,k=k,tube_nbr=4)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    #prob.check_partials(compact_print=False)
