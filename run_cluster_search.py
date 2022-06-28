# add test_things.py module to path
import sys
import numpy as np
from SimRunner import TurRunner, TauRunner, SaveSimLight, SaveParams, TurFlipper

from bimodal_dist import gauss, generate_dist
from sus.library import free_energy_probe as fep
import math

# always include these lines of MPI code!
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI procs
rank = comm.Get_rank()  # i.d. for local proc


#defaults = {'localization':18., 'location':.5, 'depth':3, 'tilt':2., 'beta':1., 'tau':1., 'scale':1., 'dt':1/10000, #'lambda':1, 'N':10_000}

initial_params = {'N':50_000,'dt':1/20_000,'target_work':None, 'tau':1, 'depth_0':4, 'depth_1':2, 'tilt':3, 'lambda':.01, 'k': np.pi**2}

local = .1
location = .4
initial_params['localization']=local
initial_params['location'] = location


def save_func(self):
   #tau = int(100*self.params['hold'])
   #local = int(self.params['localization'])
   return [f'hold{tau:03d}local{local:03d}', 'search_1/']

def decider(old_eps, new_eps, old_avg, new_avg):
   '''
   if old_avg > 6:
      if new_avg < old_avg:
         return True
      else:
         return False
   '''
   def get_prob():
      def ener(a,e):
         return 0*e**2 + ((a-3))**2/2
      return np.exp(-ener(new_avg,new_eps)/ener(old_avg,old_eps))
   
   if new_eps < old_eps or get_prob() > np.random.uniform():
      return True
   else:
      return False

#simrun.potential.domain = [[-2],[2]]

#run_name = 'heatmap_sym_detail_2X20L_5X8g_p16p31_N40/'
#save_dir = '/home/kylejray/FQ_sims/results/{}/'.format(run_name)

simrun = TurFlipper()
simrun.potential = fep.pwl_double_pot
setattr(simrun,'has_velocity',True)
simrun.change_params(initial_params)
simrun.save_procs = [SaveParams(), SaveSimLight()]


params = {}
vkey = 'lambda'
holds = np.linspace(0,.15, 4)


all_pvals = [ {vkey:hold} for hold in holds]

vals_lists = [ all_pvals[i*size:(i+1)*size] for i in range(math.ceil(len(all_pvals)/size))]
# perform local computation using local param
for vals in vals_lists:
   params = vals[rank]
   simrun.change_params(params)
   simrun.save_name = [None, None]
   # save parameters to output file by printing
   sys.stdout.write('my rank:{} of {}, params={} '.format(rank+1, size, params))
   i=0
   scaled_eps = []
   
   while i < 40:
      if i>0:
         simrun.change_params(p_current)
         simrun.perturb_params(which_params=['location', 'depth_0', 'depth_1', 'localization', 'lambda'], std=.1, n=3)


      simrun.run_sim()
      # save your results
      simrun.save_sim()
      stats = simrun.sim.output.work_stats['pos']

      if i>0:
         new_avg = stats['avg'][0]
         normal_eps = generate_dist(gauss, [np.sqrt(2*new_avg),new_avg]).get_min_eps()
         new_scaled_eps = (stats['emin'][0]-stats['tggl']) / (normal_eps-stats['tggl'])


         if decider(curr_scaled_eps, new_scaled_eps, curr_avg, new_avg) :
            scaled_eps.append([new_scaled_eps, new_avg])
            p_current = simrun.params.copy()
            curr_scaled_eps = new_scaled_eps
            curr_avg = new_avg
            print('changed params in rank{}. eps_list :'.format(rank+1),scaled_eps)
            i += 1

      else:
         curr_avg = stats['avg'][0]
         normal_eps = generate_dist(gauss, [np.sqrt(2*curr_avg),curr_avg]).get_min_eps()
         curr_scaled_eps = (stats['emin'][0]-stats['tggl']) / (normal_eps-stats['tggl'])
         scaled_eps.append([curr_scaled_eps, curr_avg])
         print('initial_run in rank{}'.format(rank+1),scaled_eps)
         p_current = simrun.params.copy()
         i += 1
      
      sys.stdout.flush()




