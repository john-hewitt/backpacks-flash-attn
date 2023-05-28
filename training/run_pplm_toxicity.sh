for iter in 0 1 2 3 4 5 6 7 8 9; do for t in toxicity; do for step in 0.00 0.02 0.04 0.06; do sbatch --account nlp --partition jag-standard --gres gpu:1 --mem 20G do_pplm_toxicity.sh $t $step; done; done; done
#for t in sports travel_adventure youth_student_life; do for step in 0.05; do sbatch --account nlp --partition jag-standard --gres gpu:1 --mem 20G do_pplm.sh $t $step; done; done
