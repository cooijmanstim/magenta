#!/bin/bash

# Assumes that this script is run from within the Coconet root directory.  Change the following line if this is not the case.
code_dir=$(dirname $0)

# Path to where generated samples were stored.
sample_file="samples/sample_20171111212415_agibbs_p01_straight_T0.99_l32_0.01min/generated_result.npy"

# Path to trained model.
checkpoint="logs/straight-64-128_bs=2,corrupt=0.5,len=64,lr=0.0625,mm=orderless,num_i=4,n_pch=46,mask_only=False,quant=0.125,rescale=True,sep=True,res=True,soft=True/best_model.ckpt"

# Evaluation settings.
#fold_index=  # Optionally can specify index of specific piece to be evaluated.
unit=frame
chronological=false
ensemble_size=5  # Number of different orderings to average.


python "$code_dir"/evaluate.py $sample_file \
--checkpoint $checkpoint \
--unit $unit \
--chronological $chronological \
--ensemble_size 5 \
#--fold_index $fold_index
