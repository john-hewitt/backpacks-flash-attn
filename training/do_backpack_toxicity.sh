#! /bin/bash
source /u/scr/johnhew/miniconda3/etc/profile.d/conda.sh
conda activate flconda-11.7
strength=$1
seed=$RANDOM
num_samples=10000
echo $topic $stepsize
cmd="python src/test_toxicity.py --toxicity_modifier $strength --seed $seed"
echo $cmd >> logs/backpack-toxicity.txt
eval $cmd
