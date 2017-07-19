"""Prepares data for basic_autofill_cnn model by blanking out pianorolls."""

import os

import numpy as np
import tensorflow as tf

import mask_tools
from pianorolls_lib import PianorollEncoderDecoder

DATASET_PARAMS = {
    'Nottingham': {
        'pitch_range': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 9},
    'MuseData': {
        'pitch_range': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 14},
    'Piano-midi.de': {
        'pitch_range': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 12,
        'batch_size': 12},
    'jsb-chorales-16th-instrs_separated': {
        'pitch_range': [36, 81], 'shortest_duration': 0.125,
        'num_instruments': 4, 'qpm': 60},
}


class DataProcessingError(Exception):
  """Exception for when data does not meet the expected requirements."""
  pass


def random_crop_pianoroll_pad(pianoroll,
                              crop_len,
                              start_crop_index=None):
  length = len(pianoroll)
  pad_length = crop_len - len(pianoroll)
  if pad_length > 0:
    pianoroll = np.pad(pianoroll, [(0, pad_length)] + [(0, 0)] * (pianoroll.ndim - 1), mode="constant")
  else:
    if start_crop_index is not None:
      start_time_idx = start_crop_index
    else:
      start_time_idx = np.random.randint(len(pianoroll) - crop_len + 1)
    pianoroll = pianoroll[start_time_idx:start_time_idx + crop_len]
  non_padded_length = length if length < crop_len else crop_len
  return pianoroll, non_padded_length


def make_data_feature_maps(sequences, hparams, encoder, start_crop_index=None):
  """Return input and output pairs of masked out and full pianorolls.

  Args:
    sequences: A list of NoteSequences.

  Returns:
    input_data: A 4D matrix with dimensions named
        (batch, time, pitch, masked_or_mask), interleaved with maskout
        pianorolls and masks.
    target: A 4D matrix of the original pianorolls with dimensions named
        (batch, time, pitch).

  Raises:
    DataProcessingError: If pianoroll is shorter than the desired crop_len, or
        if the inputs and targets have the wrong number of dimensions.
  """
  input_data = []
  targets = []
  lengths = []
  seq_count = 0
  for sequence in sequences:
    pianoroll = encoder.encode(sequence)
    cropped_pianoroll, length = random_crop_pianoroll_pad(
        pianoroll, hparams.crop_piece_len, start_crop_index)
    seq_count += 1

    # Get mask.
    T, P, I = cropped_pianoroll.shape
    unpadded_shape = length, P, I
    assert np.sum(cropped_pianoroll[length:, :, :]) == 0
    mask_fn = getattr(mask_tools, 'get_%s_mask' % hparams.maskout_method)
    mask = mask_fn(unpadded_shape,
                   separate_instruments=hparams.separate_instruments,
                   blankout_ratio=hparams.corrupt_ratio)
    if hparams.denoise_mode:
      # TODO: Denoise not yet supporting padding.
      masked_pianoroll = mask_tools.perturb_and_stack(cropped_pianoroll, mask)
    else:
      masked_pianoroll = mask_tools.apply_mask_and_stack(cropped_pianoroll, mask)

    input_data.append(masked_pianoroll)
    targets.append(cropped_pianoroll)
    lengths.append(length)
    assert len(input_data) == seq_count
    assert len(input_data) == len(targets)
    assert len(input_data) == len(lengths)

  input_data = np.asarray(input_data)
  targets = np.asarray(targets)
  lengths = np.asarray(lengths)
  if not (input_data.ndim == 4 and targets.ndim == 4):
    print input_data.ndim, targets.ndim
    raise DataProcessingError('Input data or target dimensions incorrect.')
  return input_data, targets, lengths


def get_data_as_pianorolls(basepath, hparams, fold):
  seqs, encoder = get_data_and_update_hparams(
      basepath, hparams, fold, update_hparams=False, return_encoder=True)
  assert encoder.quantization_level == hparams.quantization_level
  return [encoder.encode(seq) for seq in seqs]


def get_data_and_update_hparams(basepath, hparams, fold, 
                                update_hparams=True, 
                                return_encoder=False):
  dataset_name = hparams.dataset
  params = DATASET_PARAMS[dataset_name]
  fpath = os.path.join(basepath, dataset_name+'.npz')
  data = np.load(fpath)
  seqs = data[fold]
  pitch_range = params['pitch_range']

  if update_hparams:
    hparams.num_pitches = pitch_range[1] - pitch_range[0] + 1
    hparams.update(params)

  if not return_encoder:
    return seqs

  if params['shortest_duration'] != hparams.quantization_level:
    raise ValueError('The data has a temporal resolution of shortest '
                     'duration=%r, requested=%r' %
                     (params['shortest_duration'], hparams.quantization_level))

  encoder = PianorollEncoderDecoder(
      shortest_duration=params['shortest_duration'],
      min_pitch=pitch_range[0],
      max_pitch=pitch_range[1],
      separate_instruments=hparams.separate_instruments,
      num_instruments=hparams.num_instruments,
      encode_silences=hparams.encode_silences,
      quantization_level=hparams.quantization_level)
  return seqs, encoder
