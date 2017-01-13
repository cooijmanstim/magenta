
import os
from collections import defaultdict
import cPickle as pickle
from datetime import datetime

import numpy as np
import pylab as plt

from plotgibbs_process import pianoroll_to_midi

COLORMAP = "viridis"
COLORMAP = "bone"

# Second listening test files.
base_path = '/Users/czhuang/@coconet/compare_sampling/collect_npz'
base_path = '/Users/czhuang/@coconet/new_generation/npzs'
base_path = '/Users/czhuang/@coconet_samples/sigmoids/'
base_path = '/data/lisatmp4/huangche/sigmoids'

fpaths = {'contiguous': 'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-ContiguousMasker----schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161112185008_284.97min.npz',
          'independent': 'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-YaoSchedule-pmin-0-1--pmax-0-9--alpha-0-7---sampler-IndependentSampler-temperature-1e-05--_20161112233522_4.73min.npz',
          'nade': 'fromscratch_balanced_by_scaling_init=nade_Gibbs-num-steps-0--masker-BernoulliMasker----schedule-ConstantSchedule-1-0---sampler-SequentialSampler-temperature-1e-05--_20161112215554_5.05min.npz'}

# Yao samples for paper, 128 length, but labeled as contiguous when made the samples
fpaths = {'independent':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-500--masker-BernoulliMasker----schedule-YaoSchedule-pmin-0-1--pmax-0-9--alpha-0-7---sampler-IndependentSampler-temperature-1e-05--_20161130172135_37.83min.npz'}

# NADE samples for paper, 128 length
fpaths = {'nade':'fromscratch_balanced_by_scaling_init=nade_Gibbs-num-steps-0--masker-None--schedule-ConstantSchedule-None---sampler-SequentialSampler-temperature-1e-05--_20161201052403_37.86min.npz'}

# Sigmoid experiments, only 32 length.
fpaths = {'independent': 'fromscratch_None_init=independent_Gibbs_num_steps_848__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_0___20170108220211_12.27min.npz'}

fpaths = {'nade': 'fromscratch_None_init=sequential_Gibbs_num_steps_0__masker_None__schedule_None__sampler_None__20170108235007_1.73min.npz'}

fpaths = {'nade': 'fromscratch_None_init=sequential_Gibbs_num_steps_0__masker_None__schedule_None__sampler_None__20170109121655_24.81min.npz'}

fpaths = {'independent-1696': 'fromscratch_None_init=independent_Gibbs_num_steps_1696__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_0___20170109143550_24.55min.npz'}

fpaths = {'independent-1696-128T': 'fromscratch_None_init=independent_Gibbs_num_steps_1696__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_0___20170109162347_95.92min.npz'}

fpaths = {'sigmoid_independent_higher_temp': '/data/lisatmp4/huangche/sigmoids/fromscratch_None_init=independent_Gibbs_num_steps_20__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_0001___20170111154657_0.39min.npz'}

fpaths = {
    'independent-temp0001': 'fromscratch_None_init=independent_Gibbs_num_steps_424__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_0001___20170111160253_6.25min.npz',
    'independent-temp01': 'fromscratch_None_init=independent_Gibbs_num_steps_424__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_01___20170111160944_6.27min.npz',
    'independent-temp1': 'fromscratch_None_init=independent_Gibbs_num_steps_424__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___20170111161634_6.28min.npz'}


# Other datasets
fpaths = {
    'independent-piano-32-steps20': 'fromscratch_None_init=independent_Gibbs_num_steps_20__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___20170112143304_1.27min.npz'}

fpaths = {
    'independent-piano-32_steps20-temp01': 'fromscratch_None_init=independent_Gibbs_num_steps_20__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_1___20170112143950_1.27min.npz'}

fpaths = {
    'independent-piano-32-steps200-temp1': 'fromscratch_None_init=independent_Gibbs_num_steps_200__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___20170112144529_9.59min.npz'}
    
#STEPS_WANTED = [0, 25, 50, 75, -1]
STEPS_WANTED = range(40) + [50, 75, -1]
STEPS_WANTED = range(20) + [200/4., 200/2., 200*3/4, -1] 
#STEPS_WANTED = range(20)
STEPS_WANTED = range(0, 200, 20)
#STEPS_WANTED = [0, 424/4., 424/2., 424*3/4, -1]
NUM_SAMPLES = 4
SEPARATE_INSTRUMENTS = False


# check correct fnames.
#for method, fpath in fpaths.items():
#  assert method.lower() in fpath.lower()

# TODO: Should use the datatime from fpaths and append the key to make it more readable.
keys_str = '_'.join(fpaths.keys())
output_path = os.path.join(base_path, '%s-%s' % (keys_str, datetime.now().strftime('%Y%m%d_%H%M%S')))
os.makedirs(output_path)


def get_code(name, coding_dict):
  # hack.
  for code_key, code in coding_dict.items():
    postfix = name.split(code_key)[-1]
    if code_key in name and code != 'bach':
      return code + postfix
    elif code_key in name and code == 'bach':
      return code + postfix
  assert False, 'Match for %s was not found' % name
            

coding = {'contiguous':'c', 'independent':'i', 'nade':'n', 'bach':'b'}
method_sample_indices = defaultdict(list)
m, n = 4, 3
for i,  (method, fpath) in enumerate(fpaths.items()):
  input_fpath = os.path.join(base_path, fpath)
  print 'Loading', input_fpath
  pianoroll_steps = np.load(input_fpath)['pianorolls']
  print pianoroll_steps.shape
  assert pianoroll_steps.ndim == 5
  
  # Choose which indices in the batch to inspect. 
  random_indices = np.random.randint(100, size=NUM_SAMPLES)
  method_sample_indices[method] = random_indices
  
  #if method == 'nade':
  #  assert pianorolls.shape[0] == 1
  #  step_idx = 0
  #else:
  #  assert pianorolls.shape[0] == 101
  #  step_idx = 100
  #pianorolls = pianorolls[step_idx]
  for step in STEPS_WANTED:
    pianorolls = pianoroll_steps[step]
    print 'shape', pianorolls.shape
    #assert pianorolls.shape == (100, 32, 53, 4)
    fig, axes = plt.subplots(m, n)
    print 'axes.shape', axes.shape
    if len(str(step)) == 1:
      step_str = '0%d' % step
    else:
      step_str = '%d' % step
    
    for count_idx, idx in enumerate(random_indices):
      print method, idx
      pianoroll = pianorolls[idx].T
     
      code = get_code(method, coding)
      pp = os.path.join(
          output_path, "%s_%d_step_%s.midi" % (code, count_idx, step_str))
      print pp
      pianoroll_to_midi(pianoroll).write(pp)
      assert 0 <= pianoroll.min()
      assert pianoroll.max() <= 1
      print 'pianoroll.shape', pianoroll.shape
      if SEPARATE_INSTRUMENTS:
        assert np.allclose(pianoroll.sum(axis=1), 1)
      # max across instruments
      pianoroll = pianoroll.max(axis=0)
      ax = axes[count_idx, i]
      ax.imshow(pianoroll, cmap=COLORMAP, interpolation="none", vmin=0, vmax=1, aspect="auto", origin="lower")
      ax.set_axis_off()
      ax.set_title('%s' % method)
    #fig.suptitle("%s %i" % (method, count_idx))
    fig.set_size_inches(800 / fig.dpi, 600 / fig.dpi)
    #plt.tight_layout()
    plt.subplots_adjust(hspace=.01, wspace=.01)
#plt.show()
    plot_fpath = os.path.join(output_path, "plots-step_%s.png" % step_str)
    print 'Writing to', plot_fpath
    plt.savefig(plot_fpath, bbox_inches="tight")
    plt.close(fig)

pickle_fpath = os.path.join(output_path, 'chosen_sample_indices.pkl')
print 'Writing to', pickle_fpath
with open(pickle_fpath, 'wb') as p:
  pickle.dump(method_sample_indices, p)

text_fpath = os.path.join(output_path, 'chosen_sample_indices.txt')
print 'Writing to', text_fpath
with open(text_fpath, 'w') as p:
  p.write(str(method_sample_indices))

print 'Done'
