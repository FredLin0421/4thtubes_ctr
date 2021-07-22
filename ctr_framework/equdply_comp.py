import numpy as np
from openmdao.api import ExplicitComponent


class EqudplyComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        #self.pcd = o3d.io.read_point_cloud("/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/mesh/lumen1.PLY")
        #self.mesh = o3d.io.read_triangle_mesh("/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/mesh/lumen1.PLY")
        
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']


        #Inputs
        self.add_input('deploy_length',shape=(k,tube_nbr))
        

        # outputs
        self.add_output('equ_deploylength')


        # partials
        self.declare_partials('equ_deploylength', 'deploy_length')
        
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        deploy_length = inputs['deploy_length']
        tube_nbr = self.options['tube_nbr']
        w1 = 1.5
        w2 = 1.5
        w3 = 1
        
        'formulation 4'
        temp = w1 * (deploy_length[:,0] - deploy_length[:,1])**2 + w2 * (deploy_length[:,1] -  deploy_length[:,2])**2 \
                    # + w3 * (deploy_length[:,2] -  deploy_length[:,3])**2
        'formulation 5'
       
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.temp = temp
        # outputs['equ_deploylength'] = np.sum(equ)
        outputs['equ_deploylength'] = np.sum(temp)



    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        deploy_length = inputs['deploy_length']
        tube_nbr = self.options['tube_nbr']

        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        temp = self.temp
        pe_pd = np.zeros((k,tube_nbr))
        
        'formulation 4'
        pe_pd[:,0] = 2 * ((deploy_length[:,0] - deploy_length[:,1])) * w1 
        pe_pd[:,1] = -2 * ((deploy_length[:,0] - deploy_length[:,1])) * w1 + 2 * ((deploy_length[:,1] - deploy_length[:,2])) * w2
        pe_pd[:,2] = -2 * ((deploy_length[:,1] - deploy_length[:,2])) * w2  #
        # pe_pd[:,3] = -2 * ((deploy_length[:,2] - deploy_length[:,3])) * w3
        partials['equ_deploylength','deploy_length'][:] = pe_pd.reshape(1,-1)


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 100
    k = 1
    tube_nbr = 4
    comp = IndepVarComp()

    

    comp.add_output('deploy_length', val = np.random.rand(k,tube_nbr))
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = EqudplyComp(k=k,num_nodes=n,tube_nbr=tube_nbr)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    