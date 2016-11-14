import os, sys, time
from datetime import datetime
import numpy as np, tensorflow as tf
from magenta.models.basic_autofill_cnn import mask_tools, retrieve_model_tools, config_tools, generate_tools

import contextlib
@contextlib.contextmanager
def pdb_post_mortem():
  try:
    yield
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
      import traceback
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()

def sample_masks(shape, pm=None, k=None):
  assert (pm is None) != (k is None)
  # like mask_tools.get_random_all_time_instrument_mask except
  # produces a full batch of masks. the size of the mask follows a
  # binomial distribution, but all examples in the batch have the same
  # mask size. (this simplifies the sequential sampling logic.)
  B, T, P, I = shape
  if k is None:
    k = (np.random.rand(T * I) < pm).sum()
  print "k", k
  masks = np.zeros(shape, dtype=np.float32)
  for b in range(B):
    js = np.random.choice(T * I, size=k, replace=False)
    t = js / I
    i = js % I
    masks[b, t, :, i] = 1.
  assert np.allclose(masks.max(axis=2).sum(axis=(1,2)), k)
  return masks

def sample_contiguous_masks(shape, pm=None):
  B, T, P, I = shape
  # (deterministically) determine how many 4-timestep chunks to mask out
  chunk_size = 4
  k = int(np.ceil(pm * T * I)) // chunk_size
  masks = np.zeros(shape, dtype=np.float32)
  for b in range(B):
    ms = None
    # m * chunk_size > T would cause overlap, which would cause the
    # mask size to be smaller than desired, which breaks sequential
    # sampling.
    while ms is None or any(m * chunk_size > T for m in ms):
      if ms is not None:
        print "resampling mask to avoid overlap"
      # assign chunks to instruments
      ms = np.random.multinomial(k, pvals=[1./I] * I)
    for i in range(I):
      dt = ms[i] * chunk_size
      t = np.random.choice(T)
      t = np.arange(t, t + dt) % T
      masks[b, t, :, i] = 1.
  return masks

class BernoulliMasker(object):
  def __call__(self, shape, pm=None):
    return sample_masks(shape, pm=pm)

  def __repr__(self):
    return "BernoulliMasker()"

class ContiguousMasker(object):
  def __call__(self, shape, pm=None):
    return sample_contiguous_masks(shape, pm=pm)

  def __repr__(self):
    return "ContiguousMasker()"

class IndependentSampler(object):
  def __init__(self, temperature=1):
    self.temperature = temperature

  def __call__(self, wmodel, pianorolls, masks):
    input_data = np.asarray([
        mask_tools.apply_mask_and_stack(pianoroll, mask)
        for pianoroll, mask in zip(pianorolls, masks)])
    predictions = wmodel.sess.run(wmodel.model.predictions,
                                  {wmodel.model.input_data: input_data})
    samples = generate_tools.sample_onehot(predictions, axis=2,
                                           temperature=self.temperature)
    pianorolls = np.where(masks, samples, pianorolls)
    return pianorolls

  def __repr__(self):
    return "IndependentSampler(temperature=%r)" % self.temperature

class SequentialSampler(object):
  def __init__(self, temperature=1):
    self.temperature = temperature

  def __call__(self, wmodel, pianorolls, masks):
    B, T, P, I = pianorolls.shape

    # determine how many model evaluations we need to make
    mask_size = np.unique(masks.max(axis=2).sum(axis=(1,2)))
    # everything is better if mask sizes are the same throughout the batch
    assert mask_size.size == 1

    for _ in range(mask_size):
      input_data = np.asarray([
          mask_tools.apply_mask_and_stack(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)])
      predictions = wmodel.sess.run(wmodel.model.predictions,
                                    {wmodel.model.input_data: input_data})
      samples = generate_tools.sample_onehot(predictions, axis=2,
                                             temperature=self.temperature)

      # select one variable to sample. sample according to normalized mask;
      # is uniform as all masked out variables have equal positive weight.
      selection = masks.max(axis=2).reshape([B, T * I])
      selection = generate_tools.sample_onehot(selection, axis=1)
      selection = selection.reshape([B, T, 1, I])

      pianorolls = np.where(selection, samples, pianorolls)
      masks = np.where(selection, 0., masks)

    assert masks.sum() == 0
    return pianorolls

  def __repr__(self):
    return "SequentialSampler(temperature=%r)" % self.temperature

class BisectingSampler(object):
  def __init__(self, temperature=1):
    self.independent_sampler = IndependentSampler(temperature=temperature)

  def __call__(self, wmodel, pianorolls, masks):
    B, T, P, I = pianorolls.shape

    # determine how many variables need sampling
    mask_size = np.unique(masks.max(axis=2).sum(axis=(1,2)))
    # everything is better if mask sizes are the same throughout the batch
    assert mask_size.size == 1

    # determine bisection, a mask that is 1 where variables are to be
    # sampled independently
    bisection_size = int(np.ceil(mask_size / 2.))
    bisection = np.zeros_like(masks)
    for b in range(B):
      pm = masks[b].max(axis=1).reshape([T * I])
      pm /= pm.sum()
      js = np.random.choice(T * I, size=bisection_size, p=pm, replace=False)
      assert len(js) == bisection_size
      t = js / I
      i = js % I
      bisection[b, t, :, i] = 1.

    # sample one half independently
    ind_pianorolls = self.independent_sampler(wmodel, pianorolls, masks)
    pianorolls = np.where(bisection, ind_pianorolls, pianorolls)

    if bisection_size == mask_size:
      return pianorolls

    # sample the other half by recursive bisection
    pianorolls = self(wmodel, pianorolls, np.where(bisection, 0., masks))
    return pianorolls

  def __repr__(self):
    return "BisectingSampler(temperature=%r)" % self.independent_sampler.temperature

class YaoSchedule(object):
  def __init__(self, pmin=0.1, pmax=0.9, alpha=0.8):
    self.pmin = pmin
    self.pmax = pmax
    self.alpha = alpha

  def __call__(self, i, n):
    wat = (self.pmax - self.pmin) * i / n
    return max(self.pmin, self.pmax - wat / self.alpha)

  def __repr__(self):
    return ("YaoSchedule(pmin=%r, pmax=%r, alpha=%r)"
            % (self.pmin, self.pmax, self.alpha))

class ConstantSchedule(object):
  def __init__(self, p):
    self.p = p

  def __call__(self, i, n):
    return self.p

  def __repr__(self):
    return "ConstantSchedule(%r)" % self.p

class Gibbs(object):
  def __init__(self, num_steps, masker, sampler, schedule):
    self.num_steps = num_steps
    self.masker = masker
    self.sampler = sampler
    self.schedule = schedule

  def __call__(self, wmodel, pianorolls):
    B, T, P, I = pianorolls.shape
    for s in range(self.num_steps):
      pm = self.schedule(s, self.num_steps)
      masks = self.masker(pianorolls.shape, pm)
      pianorolls = self.sampler(wmodel, pianorolls, masks)
      assert pianorolls.sum() == B * T * I
      yield pianorolls, masks

  def __repr__(self):
    return ("Gibbs(num_steps=%r, masker=%r, schedule=%r, sampler=%r)"
            % (self.num_steps, self.masker, self.schedule, self.sampler))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model_name", None, "model name")
tf.app.flags.DEFINE_integer("num_steps", None, "number of gibbs steps to take")
tf.app.flags.DEFINE_string("sampler", None, "independent or sequential or bisecting")
tf.app.flags.DEFINE_string("masker", None, "bernoulli or contiguous")
tf.app.flags.DEFINE_string("schedule", None, "yao or constant")
tf.app.flags.DEFINE_float("schedule_yao_pmin", 0.1, "")
tf.app.flags.DEFINE_float("schedule_yao_pmax", 0.9, "")
tf.app.flags.DEFINE_float("schedule_yao_alpha", 0.7, "")
tf.app.flags.DEFINE_float("schedule_constant_p", None, "")
tf.app.flags.DEFINE_float("temperature", 1, "softmax temperature")
tf.app.flags.DEFINE_string("initialization", "random", "how to obtain initial piano roll; random or independent or nade")
# already defined in generate_tools.py
#tf.app.flags.DEFINE_string(
#    "generation_output_dir", "/Tmp/cooijmat/autofill/generate",
#    "Output directory for storing the generated Midi.")

def main(unused_argv):
  sampler = dict(
      independent=IndependentSampler,
      sequential=SequentialSampler,
      bisecting=BisectingSampler
  )[FLAGS.sampler](temperature=FLAGS.temperature)
  if FLAGS.schedule == "yao":
    schedule = YaoSchedule(pmin=FLAGS.schedule_yao_pmin,
                        pmax=FLAGS.schedule_yao_pmax,
                        alpha=FLAGS.schedule_yao_alpha)
  elif FLAGS.schedule == "constant":
    schedule = ConstantSchedule(p=FLAGS.schedule_constant_p)
  else:
    assert False
  masker = dict(bernoulli=BernoulliMasker(),
                contiguous=ContiguousMasker())[FLAGS.masker]
  gibbs = Gibbs(num_steps=FLAGS.num_steps,
                masker=masker,
                sampler=sampler,
                schedule=schedule)

  wmodel = retrieve_model_tools.retrieve_model(model_name=FLAGS.model_name)
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  start_time = time.time()

  B, T, P, I = [100, 32, 53, 4]
  if FLAGS.initialization == "random":
    pianorolls = generate_tools.sample_onehot(1 + np.random.rand(B, T, P, I), axis=2)
  elif FLAGS.initialization == "independent":
    pianorolls = np.zeros([B, T, P, I], dtype=np.float32)
    masks = np.ones_like(pianorolls)
    pianorolls = IndependentSampler(temperature=FLAGS.temperature)(wmodel, pianorolls, masks)
  elif FLAGS.initialization == "nade":
    pianorolls = np.zeros([B, T, P, I], dtype=np.float32)
    masks = np.ones_like(pianorolls)
    pianorolls = SequentialSampler(temperature=FLAGS.temperature)(wmodel, pianorolls, masks)
  else:
    assert False

  intermediates = dict(pianorolls=[pianorolls.copy()],
                       masks=[np.zeros(pianorolls.shape, dtype=np.float32)])
  for pianorolls, masks in gibbs(wmodel, pianorolls):
    intermediates["pianorolls"].append(pianorolls.copy())
    intermediates["masks"].append(masks.copy())
    sys.stderr.write(".")
    sys.stderr.flush()
  sys.stderr.write("\n")

  time_taken = (time.time() - start_time) / 60.0 #  In minutes.
  label = "".join(c if c.isalnum() else "-" for c in repr(gibbs))
  label = "fromscratch_%s_init=%s_%s_%s_%.2fmin" % (FLAGS.model_name, FLAGS.initialization, label, timestamp, time_taken)
  path = os.path.join(FLAGS.generation_output_dir, label + ".npz")
  np.savez_compressed(path, **intermediates)

if __name__ == "__main__":
  with pdb_post_mortem():
    tf.app.run()
