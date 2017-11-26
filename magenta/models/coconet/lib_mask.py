"""Tools for masking out pianorolls in different ways, such as by instrument."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import numpy as np
from magenta.models.coconet import lib_util


class MaskUseError(Exception):
  pass


def apply_mask(pianoroll, mask):
  """Apply mask to pianoroll.

  Args:
    pianoroll: A 3D binary matrix with 2D slices of pianorolls. This is not
        modified.
    mask: A 3D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
    A 3D binary matrix with masked pianoroll.

  Raises:
    MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
    raise MaskUseError('Shape mismatch in pianoroll and mask.')
  return pianoroll * (1 - mask)


def print_mask(mask):
  # assert mask is constant across pitch
  assert np.equal(mask, mask[:, 0, :][:, None, :]).all()
  # get rid of pitch dimension and transpose to get landscape orientation
  mask = mask[:, 0, :].T
  print("\n".join("".join(str({True: 1, False: 0}[z]) for z in y) for y in mask))


def get_mask(maskout_method, *args, **kwargs):
  mm = MaskoutMethod.make(maskout_method)
  return mm(*args, **kwargs)


class MaskoutMethod(lib_util.Factory):
  """Base class for mask distributions used during training."""
  pass


class BernoulliMaskoutMethod(MaskoutMethod):
  """Iid Bernoulli masking distribution."""

  key = "bernoulli"

  def __call__(self, shape, separate_instruments=True,
               blankout_ratio=0.5, **unused_kwargs):
    """Sample a mask.

    Args:
      shape: shape of pianoroll (time, pitch, instrument)
      separate_instruments: whether instruments are separated
      blankout_ratio: bernoulli inclusion probability

    Returns:
      A mask of shape `shape`.
    """
    if len(shape) != 3:
      raise ValueError(
          'Shape needs to be 3 dimensional: time, pitch, and instrument.')
    T, P, I = shape
    if separate_instruments:
      mask = np.random.random([T, 1, I]) < blankout_ratio
      mask = mask.astype(np.float32)
      mask = np.tile(mask, [1, shape[1], 1])
    else:
      mask = np.random.random([T, P, I]) < blankout_ratio
      mask = mask.astype(np.float32)
    return mask


class OrderlessMaskoutMethod(MaskoutMethod):
  """Masking distribution for orderless nade training."""

  key = "orderless"

  def __call__(self, shape, separate_instruments=True, **unused_kwargs):
    """Sample a mask.

    Args:
      shape: shape of pianoroll (time, pitch, instrument)
      separate_instruments: whether instruments are separated

    Returns:
      A mask of shape `shape`.
    """
    T, P, I = shape

    if separate_instruments:
      d = T * I
    else:
      assert I == 1
      d = T * P
    # sample a mask size
    k = np.random.choice(d) + 1
    # sample a mask of size k
    i = np.random.choice(d, size=k, replace=False)

    mask = np.zeros(d, dtype=np.float32)
    mask[i] = 1.
    if separate_instruments:
      mask = mask.reshape((T, 1, I))
      mask = np.tile(mask, [1, P, 1])
    else:
      mask = mask.reshape((T, P, 1))
    return mask
