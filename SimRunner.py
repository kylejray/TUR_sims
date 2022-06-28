import sys, os

sys.path.append(os.path.expanduser('~/source'))
from quick_sim import setup_sim
import kyle_tools as kt
import numpy as np
from scipy.stats import moment, sem
from kyle_tools.multisim import SimManager
from kyle_tools.fluctuation_theorems import ft_moment
from sus.protocol_designer import *
from sus.library.free_energy_probe import lintilt_gaussian as odw_potential
from sus.library.potentials import even_1DW

sys.path.append(os.path.expanduser('~/source/simtools/'))
# from infoenginessims.api import *
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp
from infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState


default_parameters = {'localization':18., 'location':.5, 'depth_0':3, 'depth_1':3, 'tilt':2., 'beta':1., 'tau':1., 'scale':1., 'dt':1/10000, 'lambda':1, 'N':10_000, 'target_work':1., 'k':1}



class TurRunner(SimManager):
    def __init__(self, name_func = [None, None], params=default_parameters):
        self.potential = odw_potential
        self.params = params
        self.save_name = name_func
        self.has_velocity = False

        self.save_procs = [SaveParams(), SaveSimOutput(), SaveFinalWork()]

    def verify_param(self, key, val):
        keys = list(self.params.keys())
        objectives = ['{}>0'] * len(keys)
        obj_dict = {k:v for k,v in zip(keys, objectives)}
        obj_dict['hold'] = '0 <= {} < 1'
        return eval(obj_dict[key].format(val))
        

    def initialize_sim(self):
        key_list = ['location', 'location', 'depth_0', 'depth_1', 'localization', 'localization', 'tilt']
        self.potential.default_params = [self.params[key] for key in key_list ]
        self.potential.default_params[0] *= -1
        self.potential.default_params[-1] *= 0
        #self.potential.default_params[3] = .4


        self.eq_protocol = self.potential.trivial_protocol().copy()

        self.potential.default_params[-1] = self.params['tilt']
        self.potential.default_params[2] = self.params['depth_1']
        self.potential.default_params[3] = self.params['depth_0']
        self.protocol =  self.potential.trivial_protocol().copy()

        self.system = System(self.protocol, self.potential)
        self.system.has_velocity = self.has_velocity
        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.has_velocity= self.has_velocity
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
        if not hasattr(self.sim.output, 'final_W'):
            final_state = self.sim.output.final_state
            init_state = self.sim.initial_state
            U0 = self.system.get_potential(init_state, 0) - self.eq_system.get_potential(init_state, 0)
            UF = self.eq_system.get_potential(final_state, 0) - self.system.get_potential(final_state, 0)
            final_W = U0 + UF
            setattr(self.sim.output, 'final_W', final_W)
        works =self.sim.output.final_W
        work_stats = get_work_averages(works)
        for key, value in work_stats.items():
            tur_dict = {k:v for k,v in zip(['hg','hvv','tggl'],get_turs(value['avg'][0]) )}
            value.update(tur_dict)
        work_stats['moments'] = get_moments(works, 15)
        setattr(self.sim.output, 'work_stats', work_stats)


    def set_simprocs(self, as_step):
        return [
            sp.ReturnFinalState(),
            sp.TerminateOnMean(rp.get_time_constant_work, target=self.params['target_work'], step_request=np.s_[::as_step], output_name='all_W'),
            sp.MeasureAllState(trial_request=np.s_[:200], step_request=np.s_[::as_step]), 
            tp.CountJumps(output_name='jump_trajectories'),
            ]


class SaveSimLight():
    def run(self, SimManager):
        N = SimManager.params['N']
        sim_dict = {}
        sim_dict.update({'work_stats':SimManager.sim.output.work_stats})
        sim_dict.update({'all_W':SimManager.sim.output.all_W})
        final_W = SimManager.sim.output.final_W
        fluc_hist = ft_hist(final_W)
        counts, bins = np.histogram(final_W, bins=int(2*N**(1/3)))
        sim_dict.update({'ft_hist':fluc_hist,'work_hist':[bins,counts]})
        SimManager.save_dict.update({'sim_dict':sim_dict})


class SaveParams():
    def run(self, SimManager):
        SimManager.save_dict.update({'params':SimManager.params})

class SaveSimOutput():
    def run(self, SimManager):
        keys = ['final_state', 'all_state', 'all_W', 'work_stats']
        vals = [getattr(SimManager.sim.output,item) for item in keys]
        sim_dict = { k:v for k,v in zip(keys, vals)}
        sim_dict.update({'init_state':SimManager.sim.initial_state, 'nsteps':SimManager.sim.nsteps})
        
        SimManager.save_dict.update({'sim_dict':sim_dict})

class SaveFinalWork():
    def run(self, SimManager):
        SimManager.save_dict.update({'final_W':SimManager.sim.output.final_W})

class TurFlipper(TurRunner):
    def initialize_sim(self):
        key_list = ['location', 'location', 'depth_0', 'depth_1', 'localization', 'localization', 'k']
        self.potential.default_params = [self.params[key] for key in key_list ]
        self.potential.default_params[0] *= -1
        self.potential.default_params[-1] *= 0
        #self.potential.default_params[3] = .4

        
        self.eq_protocol = self.potential.trivial_protocol().copy()

        even_1DW.default_params = [0, self.params['k']]

        self.protocol =  even_1DW.trivial_protocol().copy()

        self.system = System(self.protocol, even_1DW)
        self.system.has_velocity = self.has_velocity
        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.has_velocity= self.has_velocity
        self.system.protocol.normalize()
        self.system.protocol.time_stretch(np.pi/np.sqrt(self.params['k']))

        self.init_state = self.eq_system.eq_state(self.params['N'], t=0, beta=self.params['beta'])

        as_step = max(1, int((self.params['tau']/self.params['dt'])/500))

        self.procs = self.set_simprocs(as_step) 
        
        sim_kwargs = {'damping':self.params['lambda'], 'temp':1/self.params['beta'], 'dt':self.params['dt'], 'procedures':self.procs}
        self.sim = setup_sim(self.system, self.init_state, **sim_kwargs)
        self.sim.reference_system = self.eq_system
        return

    def verify_param(self, key, val):
        keys = list(self.params.keys())
        objectives = ['{}>0'] * len(keys)
        obj_dict = {k:v for k,v in zip(keys, objectives)}
        
        if key == 'localization':
            return 0 <= val < 2*self.params['location']
        if key == 'location':
            return  val > .5*self.params['localization']
        if key == 'depth_0':
            return 0 <= val > self.params['depth_1']
        if key == 'depth_1':
            return 0 <= val < self.params['depth_0']
        return eval(obj_dict[key].format(val))


class TauRunner(TurRunner):

    def set_simprocs(self, as_step):
        return [
            sp.ReturnFinalState(),
            sp.MeasureAllState(trial_request=np.s_[:200], step_request=np.s_[::as_step]), 
            tp.CountJumps(output_name='jump_trajectories'),
            rp.MeasureFinalValue(rp.get_dW, 'final_W'),
            rp.MeasureRunningMean(rp.get_dW, output_name='all_W', step_request=np.s_[::as_step])
            ]

    def initialize_sim(self):
        key_list = ['location', 'location', 'depth_0', 'depth_1', 'localization', 'localization', 'tilt']
        self.potential.default_params = [self.params[key] for key in key_list ]
        self.potential.default_params[0] *= -1
        self.potential.default_params[-1] *= 0

        prot = self.potential.trivial_protocol().copy()
        prot.params[-1,1] = self.params['tilt']
        prot.params[2,1] = .05*self.params['depth_0']
        rev_prot = prot.copy()
        rev_prot.reverse()
        if 'hold' in self.params.keys():
            hold_ratio = self.params['hold']
            rev_prot.time_shift(1+(2*hold_ratio)/(1-hold_ratio))
        else:
            rev_prot.time_shift(1)
        self.protocol = Compound_Protocol([prot, rev_prot])
        self.system = System(self.protocol, self.potential)
        self.system.has_velocity = self.has_velocity
        self.system.protocol.normalize()
        self.system.protocol.time_stretch(self.params['tau'])
        self.init_state = self.system.eq_state(self.params['N'], t=0, beta=self.params['beta'])
        as_step = max(1, int((self.params['tau']/self.params['dt'])/500))
        self.procs = self.set_simprocs(as_step) 

        sim_kwargs = {'damping':self.params['lambda'], 'temp':1/self.params['beta'], 'dt':self.params['dt'],    'procedures':self.procs}
        self.sim = setup_sim(self.system, self.init_state, **sim_kwargs)
        return
    

def get_work_averages(works):
    cond_keys = ['real','pos', 'neg']
    out = {key:{} for key in cond_keys}
    for c_key, cond in zip(['real','pos', 'neg'],[None, works>0, works<0]):
        temp = get_avg_stats(works, condition=cond)
        for item,key in zip(temp, ['avg','ft','emin']):
            out[c_key][key] = item
    return out

def get_turs(sigma):
    HG = 2/sigma
    HVV = 2/(np.exp(sigma)-1)
    TGGL = np.sinh(kt.inv_xtanhx(sigma/2))**-2
    try: TGGL = TGGL[0]
    except: pass
    return [HG, HVV, TGGL]

def get_moments(work, order):
    return [ft_moment(work, i+1, condition=work>0) for i in range(order)]


def get_avg_stats(work, condition=None):
    N = len(work)
    ft = np.exp(-work)
    tanh = np.tanh(work/2) 
    if condition is not None:
        cond_w = work[condition]
        Nc = len(cond_w)
        work = (Nc/N)*cond_w*(1-np.exp(-cond_w))
        ft = (Nc/N)*(1+np.exp(-cond_w))
        tanh = (Nc/N)*np.tanh(cond_w/2)*(1-np.exp(-cond_w))
        
    avg = [np.mean(work), sem(work)]
    ft = [np.mean(ft), sem(ft)]
    avg_tanh, sem_tanh = np.mean(tanh), sem(tanh)
    emin = [ 1/avg_tanh-1, (sem_tanh/avg_tanh**2)]
    return [avg, ft, emin]

def ft_hist(final_W, ax=None, nbins=None):
    W_p = final_W[final_W>0]
    W_n = final_W[final_W<0]
    if nbins is None:
        nbins = int(2*len(W_n)**(1/3))
    cp, bp = np.histogram(W_p, bins=np.linspace(0, max(W_p), nbins))
    cn, bn = np.histogram(-W_n, bins=bp)
    svals = bp[:-1]+ (bp[1]-bp[0])/2
    if ax is None:
        return [bp[:-1], cp, cn*np.exp(svals)]
    else:
        dx = bp[1]-bp[0]
        ax.bar(bp[:-1], cp, align='edge', width = dx,alpha=.5)
        ax.bar(bn[:-1], cn*np.exp(svals),align='edge', width = dx, alpha=.5)
        ax.set_yscale('log')