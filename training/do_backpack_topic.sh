#! /bin/bash
# source /u/scr/johnhew/miniconda3/etc/profile.d/conda.sh
#conda activate flconda-11.7
topic=$1
strength=$2
seed=$RANDOM
num_samples=500
echo $topic $stepsize
cmd="python src/test_topic.py --strength $strength --words_path topic_classes/${topic}.txt --seed $seed"
echo $cmd >> logs/backpack-topic_logs.txt
eval $cmd
