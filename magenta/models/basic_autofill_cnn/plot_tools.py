"""Tools for plotting training setup and results."""
from collections import namedtuple
import os

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
import tensorflow as tf

from magenta.protobuf import music_pb2

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools
from magenta.models.basic_autofill_cnn import config_tools
from magenta.lib import midi_io

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('plot_dir', '/u/huangche/plots/',
       'Path to the directory where plots will be stored.')

INSTR_ORDERING = [1, 0, 2, 3]


def get_unique_output_path():
  path = FLAGS.plot_dir
  timestamp_str = config_tools.get_current_time_as_str()
  output_path = os.path.join(path, timestamp_str)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return output_path


def get_sequence():
  fpath = os.path.join(tf.resource_loader.get_data_files_path(), 'testdata', 'generated', 'sample1.midi')
  return midi_io.midi_file_to_sequence_proto(fpath)


def get_input():
  seq = get_sequence()
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  pianoroll = encoder.encode(seq)
  return pianoroll


def set_axes_style(ax, total_time, subplots, pitch_lb=None, pitch_ub=None):
  ax.tick_params(axis=u'both', which=u'both',length=0)
  if subplots:
    labelsize=4
  else:
    labelsize=7
  ax.tick_params(axis='both', which='major', labelsize=labelsize)
  ax.tick_params(axis='both', which='minor', labelsize=labelsize)
  pitch_base = 37  # the label is actually one above 36, since starts with 0, ends at 88
  c_ticks = [12*i + 11 for i in range(4)] 
  y_ticks = [0] + c_ticks + [89-pitch_base]

  # blabel_pitch_base = pitch_base + 1  # the label is actually one above 36, since starts with 0, ends at 88
  # c_ticks = [12*i + label_pitch_base % 12 for i in range(num_pitches/12 + 1)] 
  # label_pitch_top = label_pitch_base + num_pitches
  # if label_pitch_top % 12 == 0:
  #   extra_top_tick = []
  # else:
  #   extra_top_tick = [num_pitches]
  # y_ticks = c_ticks + extra_top_tick
  # print extra_top_tick
  # print 'y_ticks', y_ticks
  ax.yaxis.set_ticks(y_ticks)
  ax.set_yticklabels([pitch_base+tick for tick in y_ticks])
  if pitch_lb is not None and pitch_ub is not None:
    ax.set_ylim([pitch_lb - pitch_base, pitch_ub - pitch_base])

  time_hop = 8
  x_ticks = range(total_time/time_hop + 1)
  ax.xaxis.set_ticks([time_hop*tick for tick in x_ticks])
  ax.set_xticklabels([time_hop*tick for tick in x_ticks])

  plt.setp(ax.spines.values(), linewidth=0.5)
  
  # Get black key positions
  b_ticks = [1, 3, 6, 8, 10]
  return c_ticks, b_ticks


def plot_summed_pianoroll(pianoroll, crop_length, ax):
  summed_pianoroll = np.clip(pianoroll[:crop_length].sum(axis=2), 0, 1)
  plt.imshow(summed_pianoroll.T, aspect='equal', cmap='Greys',
    origin='lower', interpolation='none')
  set_axes_style(ax, crop_legnth)
  plt.ylabel('pitch')
  plt.xlabel('time')


def plot_input():
  seq = get_sequence()
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  pianoroll = encoder.encode(seq)
  crop_length = 32  #64

  plt.figure(figsize=(6, 20))
  # Plot individual instruments.
  for i in INSTR_ORDERING:
    ax = plt.subplot(5, 1,  1)
    plt.imshow(pianoroll[:crop_length, :, i].T, aspect='equal', cmap='Greys',
       origin='lower', interpolation='none')
    set_axes_style(ax, crop_length)
    if i == len(INSTR_ORDERING) :
      # Only include xticks for the last subplot. 
      ax.set_xticklabels(())
    if i == 1:
      plt.ylabel('pitch')
    if i == len(INSTR_ORDERING) :
      plt.xlabel('time')
   
    # Plot all instruments combined.
    ax = plt.subplot(515)
    plot_summed_pianoroll(pianoroll, crop_length, ax)
   
    plt.savefig(os.path.join(get_unique_output_path(), 'input_vertical.png'),
       bbox_inches='tight')
    plt.close()


def plot_blankout():
  fpath = get_unique_output_path()
  # et multiple to get the one we want.
  pianoroll = get_input()
  crop_length = 32  #64
  pianoroll = pianoroll[:crop_length, :, :]
  plot_horizontal = True

  # Make colormap for target to be magenta.
  target_cmap = colors.ListedColormap(['white', 'magenta'])
  bounds = [0, 0.5, 1]
  target_norm = colors.BoundaryNorm(bounds, target_cmap.N)
  title_strs = ['Masked inputs', 'Masks', 'Targets']

  for i in range(200):
    blankouts = mask_tools.get_multiple_random_instrument_time_mask(pianoroll.shape, 4, 4)
    input_data = mask_tools.apply_mask_and_stack(pianoroll, blankouts)
    targets = pianoroll * blankouts
    print 'sum of masks:', np.sum(input_data[:, :, 4:])

  if plot_horizontal:
    # Not yet implemented for horizontal case.
    plt.figure(figsize=(13, 20))
  plt.figure(figsize=(13, 20))
  # Plot individual instruments.
  for i in INSTR_ORDERING:
    for type_ in range(len(title_strs)):
      print '\n', i, type_
      plot_idx = type_
   
     # If mask, then instr_idx should be offset by num_instruments.
    instr_idx = 0 
    if type_ == 1:
      instr_idx = pianoroll.shape[-1] 
    print 'plot_idx, instr_idx', plot_idx , instr_idx
    ax = plt.subplot(5, 3, plot_idx+1)
    if type_ in range(2):
      plt.imshow(input_data[:crop_length, :, instr_idx].T,
        aspect='equal', cmap='Greys',
        origin='lower', interpolation='none')
    else:
      plt.imshow(targets[:crop_length, :, instr_idx].T,
        interpolation='nearest', origin='lower',
        cmap=target_cmap, norm=target_norm)
    set_axes_style(ax, crop_length)
    if i == len(INSTR_ORDERING) :
      ax.set_xticklabels(())
    if i == 1:
      plt.ylabel('pitch')
    if i == len(INSTR_ORDERING) :
      plt.xlabel('time')
    if plot_idx in range(3):
      plt.title(title_strs[plot_idx])

  # Plot all instruments combined.  Just the pianoroll part.
  ax = plt.subplot(5, 3, 13)
  summed_pianoroll = np.clip(input_data[:, :, :4].sum(axis=2), 0, 1)
  plt.imshow(summed_pianoroll.T, aspect='equal', cmap='Greys',
    origin='lower', interpolation='none')
  total_time = input_data.shape[0]
  set_axes_style(ax, total_time)
  plt.ylabel('pitch')
  plt.xlabel('time')
 
  # Plot target part of all instruments combined.
  ax = plt.subplot(5, 3, 15)
  summed_pianoroll = np.clip(targets.sum(axis=2), 0, 1)
  plt.imshow(summed_pianoroll.T,
     interpolation='nearest', origin='lower',
     cmap=target_cmap, norm=target_norm)
  set_axes_style(ax, total_time)
  plt.ylabel('pitch')
  plt.xlabel('time')
 
  plt.savefig(os.path.join(fpath, 'blankouts-%d.png' ),
     bbox_inches='tight')
  plt.close()


def plot_steps(steps, original_pianoroll, output_path, run_id, 
               subplot_step_indices=None, subplots=False):
  if subplots and subplot_step_indices is None:
    raise ValueError('Need to provide subplot_step_indices')
  maskedout_instr_indices = set()
  shape = steps[0].prediction.shape
  num_timesteps, num_pitches, num_instrs = shape
  
  # TODO: just for debugging
  num_regen_iterations = len(steps) / (num_timesteps * num_instrs)
  print 'num_regen_iterations', num_regen_iterations 
  
  already_generated_pianoroll = np.zeros(shape)
  previous_already_generated_pianoroll = already_generated_pianoroll.copy()
  history_generated_pianoroll = np.zeros(shape)
  previous_change_index = None
  num_steps = len(steps)

  intermediate_seqs = []
  encoder = pianorolls_lib.PianorollEncoderDecoder()

  if not os.path.exists(output_path):
    os.mkdir(output_path)
 
  # Crop heatmap to save space:
  pitch_lb, pitch_ub = None, None

  pitch_lb = 60 # which is 59 on plot axes
  pitch_ub = 89 
  print 'shape', shape
  # The matrices for plotting is tranposed, where the first dimension is pitch.
  # Lambda function for cropping the region for plotting.
  crop_matrix_for_plot = lambda x: x  #x[:plot_pitch_start_index, :]
  shift_pitch_for_plot = lambda x: x  #x - plot_pitch_start_index
 
  for i, step in enumerate(steps):
    # Update change.
    change_index = step.change_to_context[0]
    time_step, pitch, instr_idx = change_index
    maskedout_instr_indices.add(instr_idx)
    
    # Since there might be multiple regenerations, need to blank out the pianoroll before adding.
    history_generated_pianoroll[time_step, :, instr_idx] += already_generated_pianoroll[time_step, :, instr_idx]
    already_generated_pianoroll[time_step, :, instr_idx] = 0
    already_generated_pianoroll[change_index] = 1
  
    if subplots and i not in subplot_step_indices:
      # Update previous.
      previous_already_generated_pianoroll = already_generated_pianoroll.copy()
      previous_change_index = change_index
      continue
    
    print 'plotting for step %d' % i
    # Setting up plotting.
    if subplots and i in subplot_step_indices:
      #axis = axes.flat[i]
      print '# of steps to plot:', len(subplot_step_indices)
      #plt.subplot(1, len(subplot_step_indices), subplot_step_indices.index(i)+1)
      plt.subplot(2, len(subplot_step_indices) / 2, subplot_step_indices.index(i)+1)
    else:
      plt.figure()
    axis = plt.gca()
    # TODO(annahuang): Seems to have lost the current instr original context
   
    # Mark the context that is conditioned on as darker.
    print 'maskedout_instr_indices', maskedout_instr_indices
    original_context = np.delete(original_pianoroll, list(maskedout_instr_indices), 2)
    print original_pianoroll.shape

    for t, p in zip(*np.where(crop_matrix_for_plot(original_context.sum(axis=2)))):
      axis.add_patch(Rectangle((t, p-.5), 1, 1,
           facecolor="lawngreen", edgecolor='none'))
   
    # Prediction for mask outs of current instrument.
    prediction_for_masked = step.prediction[:, :, instr_idx].T 
    # 1 (previous_already_generated_pianoroll[:, :, instr_idx].T)
    #plt.imshow(prediction_for_masked, origin='lower',
    #   nterpolation='none', aspect='auto', cmap='summer')

    # Trying log scale.
    #im = axis.imshow(20*np.log(prediction_for_masked+1), origin='lower',
    #   interpolation='none', aspect='auto', cmap='Greys') #cmap='summer')
    im = axis.imshow(crop_matrix_for_plot(prediction_for_masked), origin='lower',
      interpolation='none', aspect='equal', cmap='Greys', vmin=0, vmax=1) #cmap='summer')
   
    # Prediction on the current context.
    #predictions_for_context p.delete(
    # step.prediction.T, instr_idx, 2)
    #plt.imshow(prediction_for_masked, origin='lower',
    #   nterpolation='none', aspect='auto', cmap='summer', alpha=0.4)
   
    # Mark the historical generated as magenta boxes.
    for t, p in zip(*np.where(
        crop_matrix_for_plot(history_generated_pianoroll.sum(axis=2)))):
      axis.add_patch(Rectangle((t, p-.5), 1, 1,
           facecolor='none', edgecolor='magenta'))
   

    # Marks the already generated as magenta.
    for t, p in zip(*np.where(
        crop_matrix_for_plot(previous_already_generated_pianoroll.sum(axis=2)))):
      axis.add_patch(Rectangle((t, p-.5), 1, 1,
           facecolor='magenta', edgecolor='none'))
   
    # Mark the current instrument original.
    if instr_idx < original_pianoroll.shape[-1]:
      current_context = original_pianoroll[:, :, instr_idx]
      print current_context.shape
      for t, p in zip(*np.where(crop_matrix_for_plot(current_context))):
        axis.add_patch(Rectangle((t, p-.5), 1, 1,
             fill=None, edgecolor="lawngreen"))
   
    # Marks the current change.
    axis.add_patch(Rectangle((time_step, shift_pitch_for_plot(pitch-.5)), 1, 1,
           fill=None, edgecolor="darkturquoise", linewidth='2'))
   
    # Mark the previous change, adds a thicker border to it.
    if previous_change_index is not None:
      axis.add_patch(
          Rectangle((previous_change_index[0], shift_pitch_for_plot(previous_change_index[1]-.5)), 1, 1,
          fill=None, edgecolor="darkturquoise", linewidth='1'))

    # Setting axes styles.  
    total_time = step.prediction.shape[0]
    c_positions, b_positions = set_axes_style(
        axis, total_time, subplots, pitch_lb, pitch_ub)
 
    # Add lines for C.
    offset = 0.5
    c_positions = [c_positions[0] - 12] + c_positions
    b_positions = b_positions + [5]
    print 'c_positions', c_positions
    for c_pos in c_positions:
      c_y_pos = c_pos-offset
      c_alpha = 0.9
      print c_y_pos
      if c_pos > 0:
        plt.axhline(y=c_y_pos, xmin=0, xmax=total_time, color='royalblue', linewidth='1', alpha=c_alpha)
      # Add dash lines for black keys.
      for b_pos in b_positions:
        y_pos = c_y_pos + b_pos
        alpha = 0.15
        if b_pos == 5:
          alpha = 0.5
        else:
          y_pos += offset
        print y_pos
        plt.axhline(y=y_pos, xmin=0, xmax=total_time, color='royalblue', linewidth='1', alpha=alpha)

    # TODO: Don't know why title missing for other subplots. 
    #plt.title('Sampled step %d' % (i+1), fontsize=10)
    #plt.title(repr(step.change_to_context))
    if subplots:
      title_fontsize=6
    else:
      title_fontsize=10
    plt.title('Step %d: v%d, t%d, p%d' % (i, instr_idx, time_step, pitch+1), fontsize=title_fontsize)
    #plt.tight_layout()
    
    # TODO: Add colorbar to a different subplot..
    # TODO: can't get colorbar to position correctly in subplots
    # Setting colorbar styles.
    #if not subplots or (subplots and i + 1 == ALLOWED_STEPS):
    #  ticklabels = [0.0, 0.5, 1.0]
    #  #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=ticklabels)
    #  #fig.subplots_adjust(right=0.8)
    #  #cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    #  #cbar = fig.colorbar(im, cax=cbar_ax)
    #  cbar = plt.colorbar(im, ticks=ticklabels)
    #  cbar.ax.set_yticklabels(ticklabels, size=6)
    #  cbar.outline.set_linewidth(0.5) 
    
    if not subplots or (subplots and i == subplot_step_indices[-1]):
      fname_prefix = 'z'
      if subplots:
        fname_prefix = 'zz'
      plt.savefig(
          os.path.join(output_path, '%s_run_id_%s-iter_%d.png' % (
              fname_prefix, run_id, i)), dpi=300)#, bbox_inches='tight')
      plt.close()

    current_seq = encoder.decode(already_generated_pianoroll)
    intermediate_seqs.append(current_seq) 

    previous_already_generated_pianoroll = already_generated_pianoroll.copy()
    previous_change_index = change_index
  print 'equal', np.sum(already_generated_pianoroll), num_timesteps * num_instrs
  assert np.sum(already_generated_pianoroll) == num_timesteps * num_instrs

  return already_generated_pianoroll, intermediate_seqs


def main(argv):
  plot_input()
  plot_blankout()



if __name__ == '__main__':
  #tf.app.run()
  main(None)
