"""Classes and subroutines for generating pianorolls from coconet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from magenta.models.coconet import lib_mask
from magenta.models.coconet import lib_data
from magenta.models.coconet import lib_util
from magenta.models.coconet import lib_tfutil
from magenta.models.coconet import lib_logging


################
### Samplers ###
################

class BaseSampler(lib_util.Factory):
  """Base class for samplers.

  Samplers are callables that take pianorolls and masks and fill in the
  masked-out portion of the pianorolls.
  """

  def __init__(self, wmodel, temperature=1, logger=None, **unused_kwargs):
    """Initialize a BaseSampler instance.

    Args:
      wmodel: a WrappedModel instance
      temperature: sampling temperature
      logger: Logger instance
    """
    self.wmodel = wmodel
    self.temperature = temperature
    self.logger = logger if logger is not None else lib_logging.NoLogger()

    def predictor(pianorolls, masks):
      predictions = self.wmodel.sess.run(self.wmodel.model.predictions,
                                         {self.wmodel.model.pianorolls: pianorolls,
                                          self.wmodel.model.masks: masks})
      return predictions
    self.predictor = lib_tfutil.RobustPredictor(predictor)

  @property
  def separate_instruments(self):
    return self.wmodel.hparams.separate_instruments

  def sample_predictions(self, predictions, temperature=None):
    """Sample from model outputs."""
    temperature = self.temperature if temperature is None else temperature
    if self.separate_instruments:
      return lib_util.sample(predictions, axis=2, onehot=True,
                             temperature=temperature)
    else:
      return lib_util.sample_bernoulli(0.5 * predictions,
                                       temperature=temperature)

  @classmethod
  def __repr__(cls, self):
    return "samplers.%s" % cls.key

  def __call__(self, pianorolls, masks):
    """Sample from model.

    Args:
      pianorolls: pianorolls to populate
      masks: binary indicator of area to populate

    Returns:
      Populated pianorolls.
    """
    label = "%s_sampler" % self.key
    with lib_util.timing(label):
      return self._run_nonverbose(pianorolls, masks)

  def _run_nonverbose(self, pianorolls, masks):
    label = "%s_sampler" % self.key
    with self.logger.section(label):
      return self._run(pianorolls, masks)

class BachSampler(BaseSampler):
  """Takes Bach chorales from the validation set."""
  key = "bach"

  def _run(self, pianorolls, masks):
    if not np.all(masks):
      raise NotImplementedError()
    print("Loading validation pieces from %s..." % self.wmodel.hparams.dataset)
    dataset = lib_data.get_dataset(FLAGS.data_dir, self.wmodel.hparams, 'valid')
    bach_pianorolls = dataset.get_pianorolls()
    shape = pianorolls.shape
    pianorolls = np.array([pianoroll[:shape[1]] for pianoroll in bach_pianorolls])[:shape[0]]
    self.logger.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls

class ZeroSampler(BaseSampler):
  """Populates the pianorolls with zeros."""
  key = "zero"

  def _run(self, pianorolls, masks):
    if not np.all(masks):
      raise NotImplementedError()
    pianorolls = 0 * pianorolls
    self.logger.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls

class UniformRandomSampler(BaseSampler):
  """Populates the pianorolls with uniform random notes."""
  key = "uniform"

  def _run(self, pianorolls, masks):
    predictions = np.ones(pianorolls.shape)
    samples = self.sample_predictions(predictions, temperature=1)
    assert (samples * masks).sum() == masks.max(axis=2).sum()
    pianorolls = np.where(masks, samples, pianorolls)
    self.logger.log(pianorolls=pianorolls, masks=masks, predictions=predictions)
    return pianorolls

class IndependentSampler(BaseSampler):
  """Samples all variables independently based on a single model evaluation."""
  key = "independent"

  def _run(self, pianorolls, masks):
    predictions = self.predictor(pianorolls, masks)
    samples = self.sample_predictions(predictions)
    assert (samples * masks).sum() == masks.max(axis=2).sum()
    pianorolls = np.where(masks, samples, pianorolls)
    self.logger.log(pianorolls=pianorolls, masks=masks, predictions=predictions)
    return pianorolls

class AncestralSampler(BaseSampler):
  """Samples variables sequentially like NADE."""
  key = "ancestral"

  def __init__(self, **kwargs):
    """Initialize an AncestralSampler instance.

    Args:
      selector: an instance of BaseSelector; determines the causal order in
          which variables are to be sampled.
    """
    self.selector = kwargs.pop("selector")
    super(AncestralSampler, self).__init__(**kwargs)

  def _run(self, pianorolls, masks):
    B, T, P, I = pianorolls.shape
    assert self.separate_instruments or I == 1

    # determine how many model evaluations we need to make
    mask_size = np.max(_numbers_of_masked_variables(masks))

    with self.logger.section("sequence", subsample_factor=10):
      for _ in range(mask_size):
        predictions = self.predictor(pianorolls, masks)
        samples = self.sample_predictions(predictions)
        assert np.allclose(samples.max(axis=2), 1)
        selection = self.selector(predictions, masks,
                                  separate_instruments=self.separate_instruments)
        pianorolls = np.where(selection, samples, pianorolls)
        self.logger.log(pianorolls=pianorolls, masks=masks, predictions=predictions)
        masks = np.where(selection, 0., masks)

    self.logger.log(pianorolls=pianorolls, masks=masks)
    assert masks.sum() == 0
    return pianorolls

class GibbsSampler(BaseSampler):
  """Repeatedly resamples subsets of variables using an inner sampler."""
  key = "gibbs"

  def __init__(self, **kwargs):
    """Initialize a GibbsSampler instance.

    Args:
      masker: an instance of BaseMasker; controls how subsets are chosen.
      sampler: an instance of BaseSampler; invoked to resample subsets.
      schedule: an instance of BaseSchedule; determines the subset size.
      num_steps: number of gibbs steps to perform. If not given, defaults to
          the number of masked-out variables.
    """
    self.masker = kwargs.pop("masker")
    self.sampler = kwargs.pop("sampler")
    self.schedule = kwargs.pop("schedule")
    self.num_steps = kwargs.pop("num_steps", None)
    super(GibbsSampler, self).__init__(**kwargs)

  def _run(self, pianorolls, masks):
    B, T, P, I = pianorolls.shape
    print('shape', pianorolls.shape)
    num_steps = (np.max(_numbers_of_masked_variables(masks))
                 if self.num_steps is None else self.num_steps)
    print('num_steps', num_steps)

    with self.logger.section("sequence", subsample_factor=10):
      for s in range(num_steps):
        with lib_util.timing('gibbs step %d' %s):
          pm = self.schedule(s, num_steps)
          inner_masks = self.masker(pianorolls.shape, pm=pm, outer_masks=masks,
                                    separate_instruments=self.separate_instruments)
          pianorolls = self.sampler._run_nonverbose(pianorolls, inner_masks)
          if self.separate_instruments:
            # ensure the sampler did actually sample everything under inner_masks
            assert np.all(np.where(inner_masks.max(axis=2),
                                   np.isclose(pianorolls.max(axis=2), 1),
                                   1))
          self.logger.log(pianorolls=pianorolls, masks=inner_masks, predictions=pianorolls)

    self.logger.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls

  def __repr__(self):
    return "samplers.gibbs(masker=%r, sampler=%r)" % (self.masker, self.sampler)

class UpsamplingSampler(BaseSampler):
  """Alternates temporal upsampling and populating the gaps."""
  key = "upsampling"

  def __init__(self, **kwargs):
    self.sampler = kwargs.pop("sampler")
    self.desired_length = kwargs.pop("desired_length")
    super(UpsamplingSampler, self).__init__(**kwargs)

  def _run(self, pianorolls, masks=1.):
    if not np.all(masks):
      raise NotImplementedError()
    masks = np.ones_like(pianorolls)
    with self.logger.section("sequence"):
      while pianorolls.shape[1] < self.desired_length:
        # upsample by zero-order hold and mask out every second time step
        pianorolls = np.repeat(pianorolls, 2, axis=1)
        masks = np.repeat(masks, 2, axis=1)
        masks[:, 1::2] = 1

        with self.logger.section("context"):
          context = np.array([lib_mask.apply_mask(pianoroll, mask)
                              for pianoroll, mask in zip(pianorolls, masks)])
          self.logger.log(pianorolls=context, masks=masks, predictions=context)

        pianorolls = self.sampler(pianorolls, masks)
        masks = np.zeros_like(masks)
    return pianorolls


###############
### Maskers ###
###############

class BaseMasker(lib_util.Factory):
  """Base class for maskers."""
  @classmethod
  def __repr__(cls, self):
    return "maskers.%s" % cls.key

  def __call__(self, shape, outer_masks=1., separate_instruments=True):
    """Sample a batch of masks.

    Args:
      shape: sequence of length 4 specifying desired shape of the mask
      outer_masks: indicator of area within which to mask out
      separate_instruments: whether instruments are separated

    Returns:
      A batch of masks.
    """
    raise NotImplementedError()

class BernoulliMasker(BaseMasker):
  """Samples each element iid from a Bernoulli distribution."""
  key = "bernoulli"

  def __call__(self, shape, pm=None, outer_masks=1., separate_instruments=True):
    """Sample a batch of masks.

    Args:
      shape: sequence of length 4 specifying desired shape of the mask
      pm: Bernoulli success probability
      outer_masks: indicator of area within which to mask out
      separate_instruments: whether instruments are separated

    Returns:
      A batch of masks.
    """
    assert pm is not None
    B, T, P, I = shape
    if separate_instruments:
      probs = np.tile(np.random.random([B, T, 1, I]), [1, 1, P, 1])
    else:
      assert I == 1
      probs = np.random.random([B, T, P, I]).astype(np.float32)
    masks = probs < pm
    return masks * outer_masks

class HarmonizationMasker(BaseMasker):
  """Masks out all instruments except Soprano."""
  key = "harmonization"

  def __call__(self, shape, outer_masks=1., separate_instruments=True):
    if not separate_instruments:
      raise NotImplementedError()
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 1:] = 1.
    return masks * outer_masks

class TransitionMasker(BaseMasker):
  """Masks out the temporal middle half of the pianorolls."""
  key = "transition"

  def __call__(self, shape, outer_masks=1., separate_instruments=True):
    del separate_instruments
    masks = np.zeros(shape, dtype=np.float32)
    B, T, P, I = shape
    start = int(T * 0.25)
    end = int(T * 0.75)
    masks[:, start:end, :, :] = 1.
    return masks * outer_masks

class InstrumentMasker(BaseMasker):
  """Masks out a specific instrument."""
  key = "instrument"

  def __init__(self, instrument):
    """Initialize an InstrumentMasker instance.

    Args:
      instrument: index of instrument to mask out (S,A,T,B == 0,1,2,3)
    """
    self.instrument = instrument

  def __call__(self, shape, outer_masks=1., separate_instruments=True):
    if not separate_instruments:
      raise NotImplementedError()
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, self.instrument] = 1.
    return masks * outer_masks

#################
### Schedules ###
#################

class BaseSchedule(object):
  """Base class for Gibbs block size annealing schedule."""
  pass

class YaoSchedule(BaseSchedule):
  """Truncated linear annealing schedule.

  Please see Yao et al, https://arxiv.org/abs/1409.0585 for details."""

  def __init__(self, pmin=0.1, pmax=0.9, alpha=0.7):
    self.pmin = pmin
    self.pmax = pmax
    self.alpha = alpha

  def __call__(self, i, n):
    wat = (self.pmax - self.pmin) * i / n
    return max(self.pmin, self.pmax - wat / self.alpha)

  def __repr__(self):
    return ("YaoSchedule(pmin=%r, pmax=%r, alpha=%r)"
            % (self.pmin, self.pmax, self.alpha))

class ConstantSchedule(BaseSchedule):
  """Constant schedule."""

  def __init__(self, p):
    self.p = p

  def __call__(self, i, n):
    return self.p

  def __repr__(self):
    return "ConstantSchedule(%r)" % self.p


#################
### Selectors ###
#################

class BaseSelector(lib_util.Factory):
  """Base class for next variable selection in AncestralSampler."""

  def __call__(self, predictions, masks, separate_instruments=True):
    """Select the next variable to sample.

    Args:
      predictions: model outputs
      masks: masks within which to sample
      separate_instruments: whether instruments are separated

    Returns:
      mask indicating which variable to sample next
    """
    raise NotImplementedError()

class ChronologicalSelector(BaseSelector):
  """Selects variables in chronological order."""

  key = "chronological"

  def __call__(self, predictions, masks, separate_instruments=True):
    B, T, P, I = masks.shape
    # determine which variable to update
    if separate_instruments:
      # find index of first (t, i) with mask[:, t, :, i] == 1
      selection = np.argmax(np.transpose(masks, axes=[0, 2, 1, 3]).reshape((B, P, T * I)), axis=2)
      selection = np.transpose(np.eye(T * I)[selection].reshape((B, P, T, I)), axes=[0, 2, 1, 3])
    else:
      # find index of first (t, p) with mask[:, t, p, :] == 1
      selection = np.argmax(masks.reshape((B, T * P)), axis=1)
      selection = np.eye(T * P)[selection].reshape((B, T, P, I))
    # Intersect with mask to avoid selecting outside of the mask, e.g. in case some masks[b] is zero
    # everywhere. This can happen inside blocked Gibbs, where different examples have different
    # block sizes.
    return selection * masks

class OrderlessSelector(BaseSelector):
  """Selects variables in random order."""

  key = "orderless"

  def __call__(self, predictions, masks, separate_instruments=True):
    B, T, P, I = masks.shape
    if separate_instruments:
      # select one variable to sample. sample according to normalized mask;
      # is uniform as all masked out variables have equal positive weight.
      selection = masks.max(axis=2).reshape([B, T * I])
      selection = lib_util.sample(selection, axis=1, onehot=True)
      selection = selection.reshape([B, T, 1, I])
    else:
      selection = masks.reshape([B, T * P])
      selection = lib_util.sample(selection, axis=1, onehot=True)
      selection = selection.reshape([B, T, P, I])
    # Intersect with mask to avoid selecting outside of the mask, e.g. in case some masks[b] is zero
    # everywhere. This can happen inside blocked Gibbs, where different examples have different
    # block sizes.
    return selection * masks


def _numbers_of_masked_variables(masks, separate_instruments=True):
  if separate_instruments:
    return masks.max(axis=2).sum(axis=(1,2))
  else:
    return masks.sum(axis=(1,2,3))
