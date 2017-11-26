"""Helpers for evaluating the log likelihood of pianorolls under a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp

from magenta.models.coconet import lib_mask
from magenta.models.coconet import lib_tfutil
from magenta.models.coconet import lib_util

def evaluate(evaluator, pianorolls):
  """Evaluate a sequence of pianorolls.

  The returned dictionary contains two kinds of evaluation results: the "unit"
  losses and the "example" losses. The unit loss measures the negative
  log-likelihood of each unit (e.g. note or frame). The example loss is the
  average of the unit loss across the example. Additionally, the dictionary
  contains various aggregates such as the mean and standard error of the mean
  of both losses, as well as min/max and quartile bounds.

  Args:
    evaluator: an instance of BaseEvaluator
    pianorolls: sequence of pianorolls to evaluate

  Returns:
    A dictionary with evaluation results.
  """
  example_losses = []
  unit_losses = []

  for pi, pianoroll in enumerate(pianorolls):
    print('evaluating piece %d' % pi)
    start_time = time.time()

    unit_loss = -evaluator(pianoroll)
    example_loss = np.mean(unit_loss)

    example_losses.append(example_loss)
    unit_losses.append(unit_loss)

    duration = (time.time() - start_time) / 60.
    _report(unit_loss, prefix="%i %5.2fmin " % (pi, duration))

    if np.isinf(example_loss):
      break

  _report(example_losses, prefix="FINAL example-level ")
  _report(unit_losses, prefix="FINAL unit-level ")

  rval = dict(example_losses=example_losses,
              unit_losses=unit_losses)
  rval.update(("example_%s" % k, v) for k, v in _stats(example_losses).items())
  rval.update(("unit_%s" % k, v) for k, v in _stats(_flatcat(unit_losses)).items())
  return rval

def _report(losses, prefix=""):
  print("%s loss %s" % (prefix, _statstr(_flatcat(losses))))

def _stats(x):
  return dict(mean=np.mean(x), sem=np.std(x) / np.sqrt(len(x)),
              min=np.min(x), max=np.max(x),
              q1=np.percentile(x, 25), q2=np.percentile(x, 50), q3=np.percentile(x, 75))

def _statstr(x):
  return "mean/sem: {mean:8.5f}+-{sem:8.5f} {min:.5f} < {q1:.5f} < {q2:.5f} < {q3:.5f} < {max:.5g}".format(**_stats(x))

def _flatcat(xs):
  return np.concatenate([x.flatten() for x in xs])

class BaseEvaluator(lib_util.Factory):
  """Evaluator base class."""

  def __init__(self, wmodel, chronological):
    """Initialize BaseEvaluator instance.

    Args:
      wmodel: WrappedModel instance
      chronological: whether to evaluate in chronological order or in any order
    """
    self.wmodel = wmodel
    self.chronological = chronological

    def predictor(pianorolls, masks):
      p = self.wmodel.sess.run(
          self.wmodel.model.predictions,
          feed_dict={self.wmodel.model.pianorolls: pianorolls,
                     self.wmodel.model.masks: masks})
      return p
    self.predictor = lib_tfutil.RobustPredictor(predictor)

  @property
  def hparams(self):
    return self.wmodel.hparams

  @property
  def separate_instruments(self):
    return self.wmodel.hparams.separate_instruments

  def __call__(self, pianoroll):
    """Evaluate a single pianoroll.

    Args:
      pianoroll: a single pianoroll, shaped (T, P, I)

    Returns:
      unit losses
    """
    raise NotImplementedError()

  def _update_lls(self, lls, x, pxhat, t, d):
    """Update accumulated log-likelihoods.

    Note: the shape of `lls` and the range of `d` depends on the "number of
    variables per time step" `D`, which is the number of instruments if
    instruments if instruments are separated or the number of pitches otherwise.

    Args:
      lls: (T, D)-shaped array of unit log-likelihoods.
      x: the pianoroll being evaluated, shape (B, T, P, I).
      pxhat: the probabilities output by the model, shape (B, T, P, I).
      t: the batch of time indices being evaluated, shape (B,).
      d: the batch of variable indices being evaluated, shape (B,).
    """
    # The code below assumes x is binary, so instead of x * log(px) which is
    # inconveniently NaN if both x and log(px) are zero, we can use
    # where(x, log(px), 0).
    assert np.array_equal(x, x.astype(bool))
    index = ((np.arange(x.shape[0]), t, slice(None), d)
             if self.separate_instruments else
             (np.arange(x.shape[0]), t, d, slice(None)))
    lls[t, d] = np.log(np.where(x[index], pxhat[index], 1)).sum(axis=1)

class FrameEvaluator(BaseEvaluator):
  """Framewise evaluator.

  Evaluates pianorolls one frame at a time. That is, the model is judged for its
  prediction of entire frames at a time, conditioning on its own samples rather
  than the ground truth of other instruments/pitches in the same frame.

  The frames are evaluated in random order, and within each frame the
  instruments/pitches are evaluated in random order.
  """
  key = "frame"

  def __call__(self, pianoroll):
    T, P, I = pianoroll.shape
    assert self.separate_instruments or I == 1
    D = I if self.separate_instruments else P

    # compile a batch with each frame being an example
    B = T
    xs = np.tile(pianoroll[None], [B, 1, 1, 1])

    ts, ds = self._draw_ordering(T, D)

    # set up sequence of masks to predict the first (according to ordering)
    # instrument for each frame
    mask = []
    mask_scratch = np.ones([T, P, I], dtype=np.float32)
    for j, (t, d) in enumerate(zip(ts, ds)):
      # when time rolls over, reveal the entire current frame for purposes of
      # predicting the next one
      if j % D != 0:
        continue
      mask.append(mask_scratch.copy())
      mask_scratch[t, :, :] = 0
    assert np.allclose(mask_scratch, 0)
    del mask_scratch
    mask = np.array(mask)

    lls = np.zeros([T, D], dtype=np.float32)

    # we can't parallelize within the frame, as we need the predictions of some
    # of the other instruments. Hence we outer loop over the instruments and
    # parallelize across frames.
    xs_scratch = xs.copy()
    for d_idx in range(D):
      print('voice %d' % d_idx)
      # call the model to get predictions for the first instrument at each frame
      pxhats = self.predictor(xs_scratch, mask)

      t, d = ts[d_idx::D], ds[d_idx::D]
      assert len(t) == B and len(d) == B

      # write in predictions and update mask
      if self.separate_instruments:
        xs_scratch[np.arange(B), t, :, d] = lib_util.sample(
            pxhats[np.arange(B), t, :, d], axis=1, temperature=0, onehot=True)
        mask[np.arange(B), t, :, d] = 0
        # assert each example in the batch sees one frame more than the previous
        assert np.allclose((1 - mask).sum(axis=(1, 2, 3)),
                           [(k * D + d_idx + 1) * P
                            for k in range(mask.shape[0])])
      else:
        xs_scratch[np.arange(B), t, d, :] = lib_util.sample_bernoulli(
            pxhats[np.arange(B), t, d, :], temperature=0)
        mask[np.arange(B), t, d, :] = 0
        # assert each example in the batch sees one frame more than the previous
        assert np.allclose((1 - mask).sum(axis=(1, 2, 3)),
                           [(k * D + d_idx + 1) * I
                            for k in range(mask.shape[0])])

      self._update_lls(lls, xs, pxhats, t, d)

    # conjunction over notes within frames; frame is the unit of prediction
    return lls.sum(axis=1)

  def _draw_ordering(self, T, D):
    o = np.arange(T, dtype=np.int32)
    if not self.chronological:
      np.random.shuffle(o)
    # random variable orderings within each time step
    o = o[:, None] * D + np.arange(D, dtype=np.int32)[None, :]
    for t in range(T):
      np.random.shuffle(o[t])
    o = o.reshape([T * D])
    ts, ds = np.unravel_index(o.T, dims=(T, D))
    return ts, ds

class NoteEvaluator(BaseEvaluator):
  """Notewise evaluator.

  Evaluates pianorolls one note at a time in a randomly chosen order. Each
  prediction made by the model is conditioned on the ground truth for previous
  (according to the ordering) notes.
  """
  key = "note"

  def __call__(self, pianoroll):
    T, P, I = pianoroll.shape
    assert self.separate_instruments or I == 1
    D = I if self.separate_instruments else P
  
    # compile a batch with an example for each variable
    B = T * D
    xs = np.tile(pianoroll[None], [B, 1, 1, 1])
  
    ts, ds = self._draw_ordering(T, D)
    assert len(ts) == B and len(ds) == B
  
    # set up sequence of masks, one for each variable
    mask = []
    mask_scratch = np.ones([T, P, I], dtype=np.float32)
    for j, (t, d) in enumerate(zip(ts, ds)):
      mask.append(mask_scratch.copy())
      if self.separate_instruments:
        mask_scratch[t, :, d] = 0
      else:
        mask_scratch[t, d, :] = 0
    assert np.allclose(mask_scratch, 0)
    del mask_scratch
    mask = np.array(mask)
    
    pxhats = self.predictor(xs, mask)

    lls = np.zeros([T, D], dtype=np.float32)
    self._update_lls(lls, xs, pxhats, ts, ds)
    return lls

  def _draw_ordering(self, T, D):
    o = np.arange(T * D, dtype=np.int32)
    if not self.chronological:
      np.random.shuffle(o)
    ts, ds = np.unravel_index(o.T, dims=(T, D))
    return ts, ds

class EnsemblingEvaluator(BaseEvaluator):
  """Decorating for ensembled evaluation.

  Calls the decorated evaluator multiple times so as to evaluate according to
  multiple orderings. The likelihoods from different orderings are averaged
  in probability space, which gives a better result than averaging in log space
  (which would correspond to a geometric mean that is unnormalized and tends
  to waste probability mass).
  """
  key = "_ensembling"

  def __init__(self, evaluator, ensemble_size):
    self.evaluator = evaluator
    self.ensemble_size = ensemble_size

  def __call__(self, pianoroll):
    lls = [self.evaluator(pianoroll) for _ in range(self.ensemble_size)]
    return logsumexp(lls, b=1. / len(lls), axis=0)
