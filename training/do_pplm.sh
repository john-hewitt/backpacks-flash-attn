#! /bin/bash
# source /u/scr/johnhew/miniconda3/etc/profile.d/conda.sh
#conda activate flconda-11.7
topic=$1
stepsize=$2
seed=$RANDOM
num_samples=500
echo $topic $stepsize
cmd="python run_pplm.py -B topic_classes/${topic}.txt --cond_text None --length 100 --gamma 1.5 --num_iterations 3 --num_samples $num_samples --stepsize ${stepsize} --window_length 5 --kl_scale 0.01 --gm_scale 0.90 --top_k 0 --pretrained_model flash --colorama --sample --uncond --seed $seed  --output_path pplm-results/${topic}-gen-${stepsize}.json"
echo $cmd >> logs/pplm_logs.txt
eval $cmd
