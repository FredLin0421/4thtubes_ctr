import numpy as np
import scipy.io
import openmdao.api as om 
from openmdao.api import Problem, Group, ExecComp, IndepVarComp, ScipyOptimizeDriver, pyOptSparseDriver
# from openmdao.api import Problem, Group, ExecComp, IndepVarComp, ScipyOptimizeDriver
from ozone.api import ODEIntegrator
from ctr_framework.stiffness_comp import StiffnessComp
from ctr_framework.CtrFunction import CtrFunction
from ctr_framework.tensor_comp import TensorComp
from ctr_framework.rhs_comp import RHSComp
from ctr_framework.kinematics_comp import KinematicsComp
from ctr_framework.k_comp import KComp
from ctr_framework.sumk_comp import SumkComp
from ctr_framework.sumkm_comp import SumkmComp
from ctr_framework.invsumk_comp import InvsumkComp
from ctr_framework.tubeends_comp import TubeendsComp
from ctr_framework.initpsi_comp import InitialpsiComp
from ctr_framework.penalize_comp import PenalizeComp
from ctr_framework.interpolationkp_comp import InterpolationkpComp
from ctr_framework.straightends_comp import StraightendsComp
from ctr_framework.kappa_comp import KappaComp
from ctr_framework.kout_comp import KoutComp
from ctr_framework.interpolationkb_comp import InterpolationkbComp
from ctr_framework.interpolationkp_comp import InterpolationkpComp
'backbone comps'
from ctr_framework.backbonefunction import BackboneFunction
from ctr_framework.initR_comp import InitialRComp
from ctr_framework.u1_comp import U1Comp
from ctr_framework.u2_comp import U2Comp
from ctr_framework.u3_comp import U3Comp
from ctr_framework.u_comp import UComp
from ctr_framework.uhat_comp import UhatComp
from ctr_framework.bborientation import BborientationComp
from ctr_framework.backboneptsFunction import BackboneptsFunction
'Integrator'
from ctr_framework.finaltime_comp import FinaltimeComp
'base angle'
from ctr_framework.baseangle_comp import BaseangleComp
from ctr_framework.rotp_comp import RotpComp
from ctr_framework.baseplanar_comp import BaseplanarComp
'constraints'
from ctr_framework.diameter_comp import DiameterComp
from ctr_framework.tubeclearance_comp import TubeclearanceComp
from ctr_framework.bc_comp import BcComp
from ctr_framework.desiredpoints_comp import DesiredpointsComp
from ctr_framework.deployedlength_comp import DeployedlengthComp
from ctr_framework.beta_comp import BetaComp
from ctr_framework.pathpoints_comp import PathpointsComp
from ctr_framework.tubestraight_comp import TubestraightComp
from ctr_framework.tiporientation_comp import TiporientationComp
from ctr_framework.chi_comp import ChiComp
from ctr_framework.gamma_comp import GammaComp
from ctr_framework.kappaeq_comp import KappaeqComp
from ctr_framework.strain_comp import StrainComp
from ctr_framework.ksconstraints_comp import KSConstraintsComp
from ctr_framework.ksconstraints_min_comp import KSConstraintsMinComp
from ctr_framework.reducedimension_comp import ReducedimensionComp
from ctr_framework.strainvirtual_comp import StrainvirtualComp
'objective'
from ctr_framework.objs_comp import ObjsComp
from ctr_framework.equdply_comp import EqudplyComp
from ctr_framework.reachtargetpts_comp import ReachtargetptsComp
from ctr_framework.targetnorm_comp import TargetnormComp
from ctr_framework.jointvaluereg_comp import JointvalueregComp
from ctr_framework.locnorm_comp import LocnormComp
from ctr_framework.rotnorm_comp import RotnormComp
from ctr_framework.dp_comp import DpComp
from ctr_framework.crosssection_comp import CrosssectionComp
from ctr_framework.signedfun_comp import SignedfunComp
'mesh'
from ctr_framework.mesh import trianglemesh
'end-effector'
from ctr_framework.tippose_comp import TipposeComp
from ctr_framework.tiptransformation_comp import TiptransformationComp
from ctr_framework.endeffector_comp import EndeffectorComp
from ctr_framework.rotee_comp import RoteeComp
'orientability'
from ctr_framework.tipvec_comp import TipvecComp
from ctr_framework.normtipvec_comp import NormtipvecComp
from ctr_framework.orientability_comp import OrientabilityComp
'ozone 2'
from ctr_framework.ODE1_prob import ODE1Problem
from ctr_framework.ODE2_prob import ODE2Problem
from ctr_framework.ODE3_prob import ODE3Problem
from ctr_framework.ODE_prob import ODEProblem
from ctr_framework.timevector_comp import TimevectorComp
# from ctr_framework.objtorsion_comp import ObjtorsionComp
from ctr_framework.penalize_comp import PenalizeComp
from ctr_framework.obj_comp import ObjComp
'test'
from ctr_framework.kbkt_comp import  KbktComp
'ozone 2'
from ctr_framework.ODE1_prob import ODE1Problem
from ctr_framework.ODE2_prob import ODE2Problem
from ctr_framework.timevector_comp import TimevectorComp



class CtrseqGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', default=100, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('i')
        self.options.declare('a', default=2, types=int)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('pt')
        self.options.declare('pt_test')
        self.options.declare('target')
        self.options.declare('zeta')
        self.options.declare('rho')
        self.options.declare('eps_e')
        self.options.declare('eps_r')
        self.options.declare('eps_p')
        self.options.declare('lag')
        self.options.declare('rotx_init')
        self.options.declare('roty_init')
        self.options.declare('rotz_init')
        self.options.declare('base')
        self.options.declare('count')
        self.options.declare('equ_paras')
        self.options.declare('center')
        self.options.declare('pt_full')
        self.options.declare('viapts_nbr')
        self.options.declare('meshfile')
        # self.options.declare('des_vector')
        
        

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        i = self.options['i']
        a = self.options['a']
        equ_paras = self.options['equ_paras']
        pt = self.options['pt']
        pt_test = self.options['pt_test']
        count = self.options['count']
        zeta = self.options['zeta']
        rho = self.options['rho']
        eps_e = self.options['eps_e']
        eps_r = self.options['eps_r']
        eps_p = self.options['eps_p']
        lag = self.options['lag']
        target = self.options['target']
        tube_nbr = self.options['tube_nbr']
        rotx_init = self.options['rotx_init']
        roty_init = self.options['roty_init']
        rotz_init = self.options['rotz_init']
        base = self.options['base']
        center = self.options['center']
        pt_full = self.options['pt_full']
        viapts_nbr = self.options['viapts_nbr']
        meshfile = self.options['meshfile']
        # des_vector = self.options['des_vector']
        # mesh processing
        mesh  = trianglemesh(num_nodes,k,pt_test,center,meshfile)  
        p_ = mesh.p
        normals = mesh.normals
        
        
        # if i == 0:
            # init_guess = scipy.io.loadmat('initial.mat')
        init_guess = scipy.io.loadmat('init_2.mat')
        # elif count > 1:
        #     init_guess = scipy.io.loadmat('seq_pre'+str(count-1)+'.mat')
        # else:
            # init_guess = scipy.io.loadmat('seq_r'+str(i-1)+'.mat')
        #init_guess = scipy.io.loadmat('seq_ot3'+str(i+5)+'.mat')
        
        comp = IndepVarComp()
        comp.add_output('d1', val=init_guess['d1'])
        comp.add_output('d2', val=init_guess['d2'])
        comp.add_output('d3', val=init_guess['d3'])
        comp.add_output('d4', val=init_guess['d4'])
        comp.add_output('d5', val=init_guess['d5'])
        comp.add_output('d6', val=init_guess['d6'])
        comp.add_output('d7', val=init_guess['d7'])
        comp.add_output('d8', val=init_guess['d8'])
        comp.add_output('kappa', shape=(1,tube_nbr), val=init_guess['kappa'])
        comp.add_output('tube_section_length',shape=(1,tube_nbr),val=init_guess['tube_section_length'].T)
        comp.add_output('tube_section_straight',shape=(1,tube_nbr),val=init_guess['tube_section_straight'].T)
        comp.add_output('alpha', shape=(k,tube_nbr),val=init_guess['alpha'])
        comp.add_output('beta', shape=(k,tube_nbr),val=init_guess['beta'])
        comp.add_output('initial_condition_dpsi', shape=(k,tube_nbr), val=init_guess['initial_condition_dpsi'])
        comp.add_output('rotx',val=init_guess['rotx'])
        comp.add_output('roty',val=init_guess['roty'])
        comp.add_output('rotz',val=init_guess['rotz'])
        comp.add_output('loc',shape=(3,1),val=init_guess['loc']+1e-10)
        comp.add_output('initial_condition_p', val=np.zeros((k,3,1)))
        self.add_subsystem('input_comp', comp, promotes=['*'])
        

        # add subsystem

        'tube twist'
        stiffness_comp = StiffnessComp(tube_nbr=tube_nbr)
        self.add_subsystem('stiffness_comp', stiffness_comp, promotes=['*'])

        kbkt_comp = KbktComp(tube_nbr=tube_nbr)
        self.add_subsystem('kbkt_comp', kbkt_comp, promotes=['*'])

        tube_ends_comp = TubeendsComp(num_nodes=num_nodes,k=k,a=a,tube_nbr=tube_nbr)
        self.add_subsystem('tube_ends_comp', tube_ends_comp, promotes=['*'])
        interpolationkb_comp =  InterpolationkbComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        self.add_subsystem('interpolationkb_comp', interpolationkb_comp, promotes=['*'])
        tensor_comp = TensorComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        self.add_subsystem('tensor_comp', tensor_comp, promotes=['*'])
        sumkm_comp = SumkmComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        self.add_subsystem('sumkm_comp', sumkm_comp, promotes=['*'])
        invsumk_comp = InvsumkComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        self.add_subsystem('invsumk_comp', invsumk_comp, promotes=['*'])
        k_comp = KComp(num_nodes=num_nodes, k=k,tube_nbr=tube_nbr)
        self.add_subsystem('k_comp', k_comp, promotes=['*'])
        straightedns_comp = StraightendsComp(num_nodes=num_nodes, k=k,a=a,tube_nbr=tube_nbr)
        self.add_subsystem('straightends_comp', straightedns_comp, promotes=['*'])
        interpolationkp_comp =  InterpolationkpComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        self.add_subsystem('interpolationkp_comp', interpolationkp_comp, promotes=['*'])
        kappa_comp = KappaComp(num_nodes=num_nodes, k=k, tube_nbr = tube_nbr)
        self.add_subsystem('kappa_comp', kappa_comp, promotes=['*'])
        kout_comp = KoutComp(num_nodes=num_nodes, k=k,tube_nbr=tube_nbr)
        self.add_subsystem('kout_comp', kout_comp, promotes=['*'])
        initialpsi_comp = InitialpsiComp(num_nodes=num_nodes,k=k,tube_nbr = tube_nbr)
        self.add_subsystem('initialpsi_comp', initialpsi_comp, promotes=['*'])
        timevector_comp = TimevectorComp(num_nodes=num_nodes,tube_nbr=tube_nbr)
        self.add_subsystem('timevector_comp', timevector_comp, promotes=['*'])

        method = 'BackwardEuler'
        ODE1Problem_instance = ODE1Problem(method,'time-marching',num_times=num_nodes-1,\
                                                display='default', visualization= None)
        
        ode1 = ODE1Problem_instance.create_component()
        self.add_subsystem('integator1', ode1, promotes=['*'])

        self.add_subsystem('u1_comp', U1Comp(num_nodes = num_nodes,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u2_comp', U2Comp(num_nodes = num_nodes,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        #self.add_subsystem('u3_comp', U3Comp(num_nodes = num_nodes,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u_comp', UComp(num_nodes = num_nodes,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('u_hat', UhatComp(num_nodes = num_nodes,k=k,tube_nbr=tube_nbr), promotes = ['*'])
        self.add_subsystem('initR_comp', InitialRComp(num_nodes = num_nodes,k=k,tube_nbr=tube_nbr), promotes = ['*'])

        ODE2Problem_instance = ODE2Problem(method,'time-marching',num_times=num_nodes-1,\
                                                display='default', visualization= None)
        
        ode2 = ODE2Problem_instance.create_component()
        self.add_subsystem('integator2', ode2, promotes=['*'])
        




        
        'Transformation'

        baseanglecomp = BaseangleComp(k=k,num_nodes=num_nodes,rotx_init=rotx_init,roty_init=roty_init,rotz_init=rotz_init)
        self.add_subsystem('BaseangleComp', baseanglecomp, promotes=['*'])
        rotpcomp = RotpComp(k=k,num_nodes=num_nodes,base=base)
        self.add_subsystem('RotpComp', rotpcomp, promotes=['*'])

        '''tiposecomp = TipposeComp(k=k,num_nodes=num_nodes,tube_nbr=tube_nbr)
        self.add_subsystem('TipposeComp', tiposecomp, promotes=['*'])
        tiptransformationcomp = TiptransformationComp(k=k,num_nodes=num_nodes)
        self.add_subsystem('TiptransformationComp', tiptransformationcomp, promotes=['*'])
        endeffectorcomp = EndeffectorComp(k=k,num_nodes=num_nodes,ee_length=5)
        self.add_subsystem('EndeffectorComp', endeffectorcomp, promotes=['*'])
        Roteecomp = RoteeComp(k=k,num_nodes=num_nodes)
        self.add_subsystem('RoteeComp', Roteecomp, promotes=['*'])'''



        "Deisgn variables"
        # self.add_design_var('d1',lower= 0.2 , upper=3.5)
        # self.add_design_var('d2',lower= 0.2, upper=3.5)
        # self.add_design_var('d3',lower= 0.2, upper=3.5)
        # self.add_design_var('d4',lower= 0.2, upper=3.5)
        # self.add_design_var('d5',lower= 0.2, upper=3.5)
        # self.add_design_var('d6',lower= 0.2, upper=3.5)
        # self.add_design_var('d7',lower= 0.2, upper=3.5)
        # self.add_design_var('d8',lower= 0.2, upper=3.5)
        # tube_length_init = 0
        # tube_straight_init = 0
        # self.add_design_var('tube_section_length',lower=20,indices=[0,1,2])
        # self.add_design_var('tube_section_straight',lower=15,indices=[0,1,2])
        self.add_design_var('alpha',indices=[0,1,2])
        # temp = np.outer(np.ones(k) , -init_guess['tube_section_length']+ 2)
        self.add_design_var('beta',upper=-1,indices=[0,1,2])
        # self.add_design_var('kappa', lower=[0.001,0,0,0],upper=.1)

        self.add_design_var('initial_condition_dpsi')
        # self.add_design_var('rotx')
        # self.add_design_var('roty')
        # self.add_design_var('rotz')
        # self.add_design_var('loc')
        locnorm = LocnormComp(k=k,num_nodes=num_nodes)                                
        self.add_subsystem('LocnormComp', locnorm, promotes=['*'])
        rotnorm = RotnormComp(k=k)                                
        self.add_subsystem('rotnormComp', rotnorm, promotes=['*'])

        '''Constraints'''
        bccomp = BcComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        diametercomp = DiameterComp(tube_nbr=tube_nbr)
        tubeclearancecomp = TubeclearanceComp(tube_nbr=tube_nbr)
        tubestraightcomp = TubestraightComp(tube_nbr=tube_nbr)
        # baseplanarcomp = BaseplanarComp(num_nodes=num_nodes,k=k,equ_paras=equ_paras)
        # tiporientationcomp = TiporientationComp(k=k,tar_vector=tar_vector)
        deployedlenghtcomp = DeployedlengthComp(k=k,tube_nbr=tube_nbr)
        betacomp = BetaComp(k=k,tube_nbr=tube_nbr)

        self.add_subsystem('BetaComp', betacomp, promotes=['*'])
        #self.add_subsystem('BcComp', bccomp, promotes=['*'])
        # self.add_subsystem('Baseplanarcomp', baseplanarcomp, promotes=['*'])
        self.add_subsystem('DeployedlengthComp', deployedlenghtcomp, promotes=['*'])
        self.add_subsystem('TubestraightComp', tubestraightcomp, promotes=['*'])
        self.add_subsystem('DiameterComp', diametercomp, promotes=['*'])
        self.add_subsystem('TubeclearanceComp', tubeclearancecomp, promotes=['*'])
        # self.add_subsystem('TiporientationComp', tiporientationcomp, promotes=['*'])

        #strain
        '''num_t = 2 
        ksconstraintscomp = KSConstraintsComp(
        in_name='strain_virtual',
        out_name='strain_max',
        shape=(num_nodes,k,num_t,tube_nbr),
        axis=0,
        rho=100.,
        )
        ksconstraintsmincomp = KSConstraintsMinComp(
        in_name='strain_virtual',
        out_name='strain_min',
        shape=(num_nodes,k,num_t,tube_nbr),
        axis=0,
        rho=100.,
        )
        kappaeqcomp = KappaeqComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        gammacomp = GammaComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        chicomp = ChiComp(num_nodes=num_nodes,k=k,num_t = num_t,tube_nbr=tube_nbr)
        straincomp = StrainComp(num_nodes = num_nodes,k=k,num_t= num_t,tube_nbr=tube_nbr)
        strainvirtualcomp = StrainvirtualComp(num_nodes = num_nodes,k=k,num_t= num_t,tube_nbr=tube_nbr)
        self.add_subsystem('KappaeqComp', kappaeqcomp, promotes=['*'])
        self.add_subsystem('GammaComp', gammacomp, promotes=['*'])
        self.add_subsystem('ChiComp', chicomp, promotes=['*'])
        self.add_subsystem('StrainComp', straincomp, promotes=['*'])
        self.add_subsystem('StrainvirtualComp', strainvirtualcomp, promotes=['*'])
        self.add_subsystem('KsconstraintsComp', ksconstraintscomp, promotes=['*'])
        self.add_subsystem('KsconstraintsminComp', ksconstraintsmincomp, promotes=['*'])'''


        self.add_constraint('torsionconstraint', equals=0.)
        # self.add_constraint('tiporientation', equals=0)
        # self.add_constraint('locnorm', upper=2)
        # self.add_constraint('baseconstraints', lower=0)

        # self.add_constraint('deployedlength12constraint', lower=5)
        # self.add_constraint('deployedlength23constraint', lower=5)
        # self.add_constraint('deployedlength34constraint', lower=5)
        # self.add_constraint('deployedlength', lower=10)
        # self.add_constraint('beta12constraint', upper=-1)
        # self.add_constraint('beta23constraint', upper=-1)
        # self.add_constraint('beta34constraint', upper=-1)
        d_c = np.zeros((1,tube_nbr)) + 0.1
        # self.add_constraint('diameterconstraint',lower= d_c)
        # self.add_constraint('tubeclearanceconstraint',lower= 0.1,upper=0.16)
        # self.add_constraint('tubestraightconstraint',lower= 0)
        # self.add_constraint('strain_max',upper=0.08)
        # self.add_constraint('strain_min',lower = -0.08)
        
        
        '''objectives'''
        desiredpointscomp = DesiredpointsComp(num_nodes=num_nodes,k=k,tube_nbr=tube_nbr)
        self.add_subsystem('Desiredpointscomp', desiredpointscomp, promotes=['*'])
        reachtargetptscomp = ReachtargetptsComp(k=k,targets = pt)
        self.add_subsystem('reachtargetptsComp', reachtargetptscomp, promotes=['*'])
        targetnormcomp = TargetnormComp(k=k)
        self.add_subsystem('Targetnormcomp', targetnormcomp, promotes=['*'])
        '''dpcomp = DpComp(k=k,num_nodes=num_nodes,p_=p_,tube_nbr=tube_nbr)
        self.add_subsystem('DpComp', dpcomp, promotes=['*'])
        # crosssectioncomp = CrosssectionComp(k=k,num_nodes=num_nodes,tube_nbr=tube_nbr)
        # self.add_subsystem('CrosssectionComp', crosssectioncomp, promotes=['*'])
        signedfuncomp = SignedfunComp(k=k,num_nodes=num_nodes,normals=normals)
        self.add_subsystem('SignedfunComp', signedfuncomp, promotes=['*'])
        equdply = EqudplyComp(k=k,num_nodes=num_nodes,tube_nbr=tube_nbr)
        self.add_subsystem('EqudplyComp', equdply, promotes=['*'])'''

        # orientability
        
        # tipveccomp = TipvecComp(k=k,tube_nbr=tube_nbr,num_nodes=num_nodes)
        # self.add_subsystem('TipvecComp', tipveccomp, promotes=['*'])
        # normtipveccomp = NormtipvecComp(k=k,tube_nbr=tube_nbr,num_nodes=num_nodes)
        # self.add_subsystem('Normtipveccomp', normtipveccomp, promotes=['*'])
        # orientabilitycomp = OrientabilityComp(k=k,num_nodes=num_nodes,des_vector=init_guess['des_vector'])
        # self.add_subsystem('OrientabilityComp', orientabilitycomp, promotes=['*'])

        # objective function
        dl0 = init_guess['tube_section_length'].T + init_guess['beta']
        norm1 = np.linalg.norm(pt_full[0,:]-pt_full[-1,:],ord=1.125)
        norm2 = (dl0[:,0] - dl0[:,1])**2 + (dl0[:,1] -  dl0[:,2])**2
        norm3 = np.linalg.norm(pt_full[0,:]-pt_full[-1,:])/viapts_nbr
        norm4 = 2
        norm5 = 2*np.pi
        eps_o = 20 * 2.5

        # objscomp = ObjsComp(k=k,num_nodes=num_nodes,
        #                     zeta=zeta,
        #                         rho=rho,
        #                             eps_r=eps_r,
        #                                 eps_p=eps_p,
        #                                     lag=lag,
        #                                         norm1 = norm1,
        #                                             norm2 = norm2,
        #                                                 norm3 = norm3,
        #                                                     norm4 = norm4,
        #                                                         norm5 = norm5,
        #                                                             eps_e = eps_e,
        #                                                                 eps_o = eps_o)                                    
        # self.add_subsystem('ObjsComp', objscomp, promotes=['*'])
        # self.add_objective('objs')
        # objtorsioncomp = ObjtorsionComp(k=k,tube_nbr=tube_nbr,num_nodes=num_nodes)

        # self.add_subsystem('ObjtorsionComp', objtorsioncomp, promotes=['*'])
        # self.add_objective('objtorsion')

        # penalizecomp = PenalizeComp(k=k,tube_nbr=tube_nbr,num_nodes=num_nodes)
        # self.add_subsystem('PenalizeComp', penalizecomp, promotes=['*'])
        # objcomp = ObjComp(k=k,tube_nbr=tube_nbr,num_nodes=num_nodes)
        # self.add_subsystem('ObjComp', objcomp, promotes=['*'])
        self.add_objective('targetnorm')
        