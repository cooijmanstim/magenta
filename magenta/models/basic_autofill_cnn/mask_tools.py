"""Tools for masking out pianorolls in different ways, such as by instrument."""


 
import numpy as np


class MaskUseError(Exception):
  pass


def apply_mask_and_interleave(pianoroll, mask):
  """Depth concatenate pianorolls and masks by interleaving them.

  Args:
    pianoroll: A 3D binary matrix with 2D slices of pianorolls. This is not
        modified.
    mask: A 3D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
    A 3D binary matrix with masked pianoroll and masks interleaved.

  Raises:
    MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
    raise MaskUseError('Shape mismatch in pianoroll and mask.')
  masked_pianoroll = pianoroll * (1 - mask)
  timesteps, pitch_range, num_instruments = masked_pianoroll.shape
  pianoroll_and_mask = np.zeros(
      (timesteps, pitch_range, num_instruments * 2), dtype=np.float32)
  for instr_idx in range(num_instruments):
    pianoroll_and_mask[:, :, instr_idx * 2] = masked_pianoroll[:, :, instr_idx]
    pianoroll_and_mask[:, :, instr_idx * 2 + 1] = mask[:, :, instr_idx]
  return pianoroll_and_mask


def apply_mask_and_stack(pianoroll, mask):
  """Stack pianorolls and masks on the last dimension.

  Args:
    pianoroll: A 3D binary matrix with 2D slices of pianorolls. This is not
        modified.
    mask: A 3D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
    A 3D binary matrix with masked pianoroll and mask stacked.

  Raises:
    MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
    raise MaskUseError('Shape mismatch in pianoroll and mask.')
  masked_pianoroll = pianoroll * (1 - mask)
  return np.concatenate([masked_pianoroll, mask], 2)


def get_random_instrument_mask(pianoroll_shape):
  """Creates a mask to mask out a random instrument.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.

  Returns:
    A 3D binary mask.
  """
  instr_idx = np.random.randint(pianoroll_shape[-1])
  return get_instrument_mask(pianoroll_shape, instr_idx)


def get_instrument_mask(pianoroll_shape, instr_idx):
  """Creates a mask to mask out the instrument at given index.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    instr_idx: An integer index indicating which instrument to be masked out.

  Returns:
    A 3D binary mask.
  """
  mask = np.zeros((pianoroll_shape))
  mask[:, :, instr_idx] = np.ones(pianoroll_shape[:2])
  return mask


def get_multiple_random_patch_mask(pianoroll_shape, mask_border,
                                   initial_maskout_factor):
  """Creates a mask with multiple random patches to be masked out.

  This function first randomly selects locations in the pianoroll. The number
  of such selections is given by the initial_maskout_factor * size of pianoroll.
  The masked patches are then the bordered square patches around the selections.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border of the mask in number of cells.
    initial_maskout_factor: The initial percentage of how much mask locations to
        generate.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape, dtype=np.bool)
  random_inds = np.random.permutation(mask.size)
  num_initial_blankouts = int(np.ceil(mask.size * initial_maskout_factor))
  blankout_inds = np.unravel_index(random_inds[:num_initial_blankouts],
                                   mask.shape)
  mask[blankout_inds] = 1
  # Set up a different mask for each instrument.
  for instr_idx in range(pianoroll_shape[-1]):
    for axis in [0, 1]:
      # Shift the mask to make sure some longer notes are blanked out.
      for shifts in range(-mask_border, mask_border + 1):
        mask[:, :, instr_idx] += np.roll(
            mask[:, :, instr_idx], shifts, axis=axis)
  return mask.astype(np.float32)


def get_random_pitch_range_mask(pianoroll_shape, mask_border):
  """Creates a mask to mask out all time steps in a random pitch range.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border below and above a randomly choosen pitch for masking
        out a range of pitches.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  _, pitch_range, num_instruments = pianoroll_shape
  instr_idx = np.random.randint(num_instruments)
  random_pitch_center = np.random.randint(pitch_range)
  upper_pitch = random_pitch_center + mask_border + 1
  lower_pitch = random_pitch_center - mask_border
  if lower_pitch < 0:
    lower_pitch = 0
  mask[:, lower_pitch:upper_pitch, instr_idx] = 1
  return mask


def get_random_time_range_mask(pianoroll_shape, mask_border):
  """Mask out all notes in a random time range across all pitches.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  time_range, _, num_instruments = pianoroll_shape
  # Mask out only one intrument.
  instr_idx = np.random.randint(num_instruments)
  random_time_center = np.random.randint(time_range)
  for time_shift in range(-mask_border, mask_border):
    time_idx = (time_shift + random_time_center) % time_range
    mask[time_idx, :, instr_idx] = 1
  return mask


def get_random_instrument_time_mask(pianoroll_shape, timesteps, voices_for_mask_candidate=None):
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  time_range, _, num_instruments = pianoroll_shape
  # Mask out only one intrument.
  if voices_for_mask_candidate is None:
    voices_for_mask_candidate = range(num_instruments)
  instr_idx = np.random.choice(voices_for_mask_candidate)
  random_start_idx = np.random.randint(time_range)
  end_idx = random_start_idx + timesteps
  #print 'random_start_idx, end_idx', random_start_idx, end_idx 
  for time_idx in range(random_start_idx, end_idx):
    time_idx %= time_range
    mask[time_idx, :, instr_idx] = 1
  assert np.sum(mask) != timesteps
  return mask


def get_multiple_random_instrument_time_mask_by_mask_size(pianoroll_shape, mask_size, 
                                             num_maskout, voices_for_mask_candidate=None):
  """Mask out multiple random time ranges, randomly across instruments.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  for i in range(num_maskout):
    mask += get_random_instrument_time_mask(pianoroll_shape, mask_size, voices_for_mask_candidate)
  return np.clip(mask, 0, 1)


def get_multiple_random_instrument_time_mask(pianoroll_shape, mask_border,
                                             num_maskout, voices_for_mask_candidate=None):
  """Mask out multiple random time ranges, randomly across instruments.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  for i in range(num_maskout):
    mask += get_random_time_range_mask(pianoroll_shape, mask_border)
  return np.clip(mask, 0, 1)


def get_multiple_random_instrument_time_mask_next(pianoroll_shape, mask_border,
                                                  num_maskout):
  """Mask out middle of pianoroll, across all instruments.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  num_timesteps = pianoroll_shape[0]
  one_fourth_duration = num_timesteps / 4
  end_index = num_timesteps - one_fourth_duration
  mask[one_fourth_duration:end_index, :, :] = 1
  return mask


def get_distribution_on_num_diff_instr():
  """Check distribution of number of instruments masked in random selection."""
  num_tries = 1000
  num_instrs = []
  for i in range(num_tries):
    instrs = set()
    for j in range(4):
      instrs.add(np.random.randint(4))
    num_instrs.append(len(instrs))
  hist = np.histogram(num_instrs, bins=range(1, 5))
  return hist
