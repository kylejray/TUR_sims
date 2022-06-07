import sys, os

sys.path.append(os.path.expanduser('~/source'))
from quick_sim import setup_sim
import kyle_tools as kt
from kyle_tools.multisim import SimManager
from sus.protocol_designer import *
from sus.library.free_energy_probe import lintilt_gaussian as odw_potential

sys.path.append(os.path.expanduser('~/source/simtools/'))
# from infoenginessims.api import *
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp
from infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState


default_parameters = {'localization':18., 'location':.5, 'depth':3, 'tilt':2., 'beta':1., 'tau':1., 'scale':1., 'dt':1/10000, 'lambda':1, 'N':10_000, 'target_work':1.}



class TurRunner(SimManager):
    def __init__(self, name_func = [None, None], params=default_parameters):
        self.potential = odw_potential
        self.params = params
        self.save_name = name_func

        self.save_procs = [SaveParams(), SaveSimOutput(), SaveFinalWork()]

    
    def initialize_sim(self):
        key_list = ['location', 'location', 'depth', 'depth', 'localization', 'localization']
        self.potential.default_params = [self.params[key] for key in key_list ]
        self.potential.default_params[0] *= -1
        self.eq_protocol = self.potential.trivial_protocol().copy()

        self.potential.default_params[2] -= self.params['tilt']
        self.potential.default_params[3] += self.params['tilt']
        self.protocol =  self.potential.trivial_protocol().copy()

        self.system = System(self.protocol, self.potential)
        self.system.has_velocity=False
        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.has_velocity=False
        self.system.protocol.normalize()
        self.system.protocol.time_stretch(self.params['tau'])

        self.init_state = self.eq_system.eq_state(self.params['N'], t=0, beta=self.params['beta'])

        as_step = max(1, int((self.params['tau']/self.params['dt'])/500))

        self.procs = self.set_simprocs(as_step) 
        
        sim_kwargs = {'damping':self.params['lambda'], 'temp':1/self.params['beta'], 'dt':self.params['dt'], 'procedures':self.procs}
        self.sim = setup_sim(self.system, self.init_state, **sim_kwargs)
        self.sim.reference_system = self.eq_system
        return
    
    def analyze_output(self):
        final_state = self.sim.output.final_state
        init_state = self.sim.initial_state
        U0 = self.system.get_potential(init_state, 0) - self.eq_system.get_potential(init_state, 0)
        UF = self.eq_system.get_potential(final_state, 0) - self.system.get_potential(final_state, 0)
        final_W = U0 + UF

        setattr(self.sim.output, 'final_W', final_W)

    def set_simprocs(self, as_step):
        return [
            sp.ReturnFinalState(),
            sp.TerminateOnMean(rp.get_time_constant_work, target=self.params['target_work'], step_request=np.s_[::as_step], output_name='all_W'),
            sp.MeasureAllState(trial_request=np.s_[:200], step_request=np.s_[::as_step]), 
            tp.CountJumps(output_name='jump_trajectories'),
            ]
    
class SaveParams():
    def run(self, SimManager):
        SimManager.save_dict.update({'params':SimManager.params})

class SaveSimOutput():
    def run(self, SimManager):
        keys = ['final_state', 'all_state', 'all_W']
        vals = [getattr(SimManager.sim.output,item) for item in keys]
        sim_dict = { k:v for k,v in zip(keys, vals)}
        sim_dict.update({'init_state':SimManager.sim.initial_state, 'nsteps':SimManager.sim.nsteps})
        
        SimManager.save_dict.update({'sim_dict':sim_dict})

class SaveFinalWork():
    def run(self, SimManager):
        SimManager.save_dict.update({'final_W':SimManager.sim.output.final_W})






