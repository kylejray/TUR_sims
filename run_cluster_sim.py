# add test_things.py module to path
import sys
import numpy as np
from SimRunner import TurRunner

# always include these lines of MPI code!
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI procs
rank = comm.Get_rank()  # i.d. for local proc


defaults = {'localization':18., 'location':.5, 'depth':3, 'tilt':2., 'beta':1., 'tau':1., 'scale':1., 'dt':1/10000, 'lambda':1, 'N':10_000}

N=100_000
dt = 1/20_000

def save_func(self):
   N = int(self.params['N']/1000)
   tau = int(100*self.params['tau'])
   return [f'N{N}tau{tau:03d}', None]

simrun = TurRunner(params = defaults)
simrun.save_name = save_func
simrun.change_params({'N':N,'dt':dt})

taus= [1]

#run_name = 'heatmap_sym_detail_2X20L_5X8g_p16p31_N40/'
#save_dir = '/home/kylejray/FQ_sims/results/{}/'.format(run_name)

#max_L = 8
#L_lists = [ L_range[i*max_L:(i+1)*max_L] for i in range(math.ceil(len(L_range)/max_L))]


# my_param = params[rank]
time = taus[rank]
simrun.change_params({'tau': time})
# save parameters to output file by printing
sys.stdout.write('my rank:{} of {}, tau={} '.format(rank+1, size, simrun.params['tau']))
# perform local computation using local param
simrun.run_sim()
# save your results
simrun.save_sim()

sys.stdout.flush()
