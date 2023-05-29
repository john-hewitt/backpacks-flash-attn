# Backpack Language Models (ACL 2023)

This repository provides the code necessary to replicate the paper
_[Backpack Language Models](https://arxiv.org/abs/2305.16765)_, the ACL 2023 paper. This includes both
the training of the models and the subsequent evaluation.

This is not necessarily the best way to use Backpacks.
You may instead be looking for:

 - The [Huggingface Transformers model code](https://huggingface.co/stanfordnlp/backpack-gpt2) that implements Backpacks.
 - The [Backpack Models website](https://backpackmodels.science)

This codebase is a fork of the [FlashAttention](https://github.com/HazyResearch/flash-attention/) codebase, which has
had some breaking changes since the fork. Thus, you'll need to install
_this_ FlashAttention version.



## Using this codebase

### Install

#### Note on conda install
I had a hard time installing FlashAttention the first time, including a lot of manual
work to get the CUDA kernels to install and problems matching up the CUDA version to
Pytorch, so here's a description of what I did when replicating on GCP:

    conda create --name flenv
    conda activate flenv
    conda install python==3.9.0
    sudo apt-get install cuda-11.7 # CUDA version is important!


At this point I downloaded the `switch-cuda.sh` [script](https://github.com/phohenecker/switch-cuda), which if nothing else helped
me understand what CUDA version was going to be used (when multiple are installed.)
I placed it at `~/switch-cuda.sh` and ran:

    source ~/switch-cuda.sh 11.7
    pip install torch==1.13.1+cu117   --extra-index-url https://download.pytorch.org/whl/cu117 # CUDA version is important!
    pip install --no-cache-dir numpy==1.23.5
    pip install --verbose -e . # To be clear, inside the top-level directory of this repository.
    # Verbose flag up there is just because this takes forever.
    pip install transformers # 4.27 works
    pip install hydra-core --upgrade
    conda install rich
    pip install pytorch-lightning==1.8.1
    pip install transformers
    pip install datasets
    pip install matplotlib
    pip install seaborn
    pip install evaluate stanza
    pip install scipy
    conda install -c pytorch faiss-cpu
    pip install scikit-learn mauve-text
    pip install python-dotenv
    pip install hydra-colorlog
    conda install apex


From the `training/` directory, run:

    # Installing CUDA kernels
    cd ../csrc/fused_dense_lib && pip install . && cd -
    cd ../csrc/xentropy && pip install . && cd -
    cd ../csrc/rotary && pip install . && cd -
    cd ../csrc/layer_norm && pip install .

Every time I run experiments, I do the following:

    export PYTHONPATH=$PWD:$PYTHONPATH
    source ~/switch-cuda.sh 11.7
    conda activate flenv



#### Install FlashAttention

Install FlashAttention via the `flash_attn_README.md` instructions in this
folder. It has been modified slightly to install from the local code instead
of the most recent `pip` version.

Also follow the installation instructions in `training/flash_attn_README.md`.

### (Optional) Train your own Backpack LM on OpenWebText.

If you want to replicate the whole process of the paper from scratch, you'll
need to pretrain your own Backpack Language Model! It's not too involved,
as the FlashAttention codebase is quite good, and the models we worked with
are rather modest in size (maximum 170M parameters.) I trained them on a set
of 4 NVIDIA A100 (40G) GPUs, and it took around 3 or 4 days.

Follow the instructions in `training/README.md` to build the OpenWebText
dataset (under **Dataset preparation**.) Then run

```
    torchrun --nproc_per_node=4 run.py experiment=owt/backpack-micro-flash trainer.devices=4 name=backpack-micro-flash-fp16 datamodule.batch_size=128
    torchrun --nproc_per_node=4 run.py experiment=owt/backpack-mini-flash trainer.devices=4 name=backpack-mini-flash-fp16 datamodule.batch_size=128
    torchrun --nproc_per_node=4 run.py experiment=owt/backpack-small-flash trainer.devices=4 name=backpack-small-flash-fp16 datamodule.batch_size=128
```

### (Optional) Download a pretrained Backpack LM.

Don't want to train your own Backpack? Understood. All models trained for the paper are available (Backpacks as well as Transformers for comparison.)

#### Main models.
These are the reference Backpack language models; the `Small` model is the model evaluated throughout the _Backpack Language Models_ paper.

|     | Backpack  | Transformer |
| --- | ---------| -----------|
| Micro | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/main/backpack-micro-flash-fp16.ckpt)   | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/main/gpt2-micro-flash-fp16.ckpt)     |
| Mini | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/main/backpack-mini-flash-fp16.ckpt)   | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/main/gpt2-mini-flash-fp16.ckpt)     |
| Small| [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/main/backpack-small-flash-fp16.ckpt)   | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/main/gpt2-small-flash-fp16.ckpt)     |


#### Sense count ablation models
These models were used in our ablation studying the effect of the number of sense vectors.
They are different from the models above not just in number of sense vectors and less training,
, but also in that they have far fewer parameters in their sense vector networks, so they should
be avoided in favor of the models above.

|     | Backpack |
| --- | ---------|
| 1   | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/sense_ablation/backpack-mini-flash-fp16-vecs-1.ckpt)   |
| 4   | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/sense_ablation/backpack-mini-flash-fp16-vecs-4.ckpt)   |
| 16  | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/sense_ablation/backpack-mini-flash-fp16-vecs-16.ckpt)   |
| 64  | [(link)](https://downloads.cs.stanford.edu/nlp/data/backpacks-acl2023/sense_ablation/backpack-mini-flash-fp16-vecs-64.ckpt)   |

## Experiment Replication

### LM Evaluation Experiments (Section 4)

These experiments are somewhat unalike the rest in that they're run through the
Eleuther AI LM evaluation harness. So, you'll need that!

We modified it a bit to allow simple loading of our models, so you'll need our fork:

    cd ..
    git clone https://github.com/john-hewitt/lm-evaluation-harness

And run the installation specified there.

Then, run the following

    cd lm-evaluation-harness
    bash do_all.sh

The path to the checkpoint is currently hard-coded into line `59` of `lm_eval/models/gpt2.py`,
so we need to fix that. But pointing it at your checkpoint, it should work!

### Emergent Structure in Sense Vectors (Section 5)

All subsequent experiments should be run in the `training` directory. So,

    cd training

Now, we need to download all the lexical similarity datasets.

Now, we can run the lexical similarity evaluations

    # Backpacks with sense vectors
    python src/run_simlex.py  --checkpoint checkpoints/backpack-small-flash-fp16/last.ckpt --use_softmax 0 --multivec_methods 1
    # Backpacks with Embedding matrix
    python src/run_simlex.py  --checkpoint checkpoints/backpack-small-flash-fp16/last.ckpt
    # Transformers with Embedding matrix
    python src/run_simlex.py  --checkpoint checkpoints/gpt2s-small-flash-fp16/last.ckpt 

    # Other LMs
    python src/run_simlex.py  --checkpoint gpt2
    python src/run_simlex.py  --checkpoint gpt2-medium
    python src/run_simlex.py  --checkpoint gpt2-large
    python src/run_simlex.py  --checkpoint gpt2-xl
    python src/run_simlex.py  --checkpoint EleutherAI/gpt-j-6B --use_first 1

### Control using Sense Vectors (Section 6)

#### Topic control
The topic control experiments 

Make some directories

    mkdir logs
    mkdir backpack-topic-results

Running Backpack sense vector topic experiments (I run these independently on slurm.)

    for t in arts_culture business_entrepreneurs celebrity_pop_culture diaries_daily_life family fashion_style film_tv_video fitness_health food_dining gaming music news_social_concern other_hobbies relationships sports travel_adventure youth_student_life; do for strength in 0 1 2 3; do bash do_backpack_topic.sh $t $strength ; done; done

To get the semantic success averages for Backpacks across all topics, run

    # For "0" (no semantic control)
    cat logs/backpack*_0__topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"
    # For "1" (little semantic control)
    cat logs/backpack*_1__topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"
    # For "2" (more semantic control)
    cat logs/backpack*_2__topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"
    # For "3" (most semantic control)
    cat logs/backpack*_3__topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"

Now, combine the results (for MAUVE computation)

    cd backpack-topic-results
    python combine_all.py all-0.json *-0
    python combine_all.py all-1.json *-1
    python combine_all.py all-2.json *-2
    python combine_all.py all-3.json *-3

And run MAUVE calculation to get the MAUVE scores in Figure 2 and Table 11.

    cd ..
    python src/run_mauve.py --refs val-100len.json --preds backpack-topic-results/all-0.json
    python src/run_mauve.py --refs val-100len.json --preds backpack-topic-results/all-1.json
    python src/run_mauve.py --refs val-100len.json --preds backpack-topic-results/all-2.json
    python src/run_mauve.py --refs val-100len.json --preds backpack-topic-results/all-3.json

Running PPLM experiments. This takes a much longer time than generation with Backpacks, above.
I highly suggest paralellizing this across jobs.
The hyperparameter varied is the stepsize.

    for t in arts_culture business_entrepreneurs celebrity_pop_culture diaries_daily_life family fashion_style film_tv_video fitness_health food_dining gaming music news_social_concern other_hobbies relationships sports travel_adventure youth_student_life; do for step in 0.00 0.01 0.04 0.05; do bash do_pplm.sh $t $step; done; done

Running the semantic success evaluations on the PPLM results (These were run automatically for the Backpacks during generation.).

    for t in arts_culture business_entrepreneurs celebrity_pop_culture diaries_daily_life family fashion_style film_tv_video fitness_health food_dining gaming music news_social_concern other_hobbies relationships sports travel_adventure youth_student_life; do python src/test_topic.py --generations_path pplm-results/$t-gen-0.00.json ; done
    for t in arts_culture business_entrepreneurs celebrity_pop_culture diaries_daily_life family fashion_style film_tv_video fitness_health food_dining gaming music news_social_concern other_hobbies relationships sports travel_adventure youth_student_life; do python src/test_topic.py --generations_path pplm-results/$t-gen-0.01.json ; done
    for t in arts_culture business_entrepreneurs celebrity_pop_culture diaries_daily_life family fashion_style film_tv_video fitness_health food_dining gaming music news_social_concern other_hobbies relationships sports travel_adventure youth_student_life; do python src/test_topic.py --generations_path pplm-results/$t-gen-0.04.json ; done
    for t in arts_culture business_entrepreneurs celebrity_pop_culture diaries_daily_life family fashion_style film_tv_video fitness_health food_dining gaming music news_social_concern other_hobbies relationships sports travel_adventure youth_student_life; do python src/test_topic.py --generations_path pplm-results/$t-gen-0.05.json ; done

To get the semantic success averages for PPLM across all topics, run

    # For "0.00" (no semantic control)
    cat logs/pplm*0.00.json-topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"
    # For "0.01" (no semantic control)
    cat logs/pplm*0.01.json-topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"
    # For "0.04" (more semantic control)
    cat logs/pplm*0.04.json-topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"
    # For "0.05" (most semantic control)
    cat logs/pplm*0.05.json-topic_summary | grep "Sem" | cut -f 2 -d' ' | python -c "import sys; print((lambda x: sum(x)/len(x))([float(x.strip()) for x in sys.stdin]))"

Now we combine together the PPLM generations for MAUVE computation.

    python backpack-topic-results/combine_all.py pplm-results/all-0.00.json pplm-results/*-0.00
    python backpack-topic-results/combine_all.py pplm-results/all-0.01.json pplm-results/*-0.01
    python backpack-topic-results/combine_all.py pplm-results/all-0.04.json pplm-results/*-0.04
    python backpack-topic-results/combine_all.py pplm-results/all-0.05.json pplm-results/*-0.05

And finally, compute MAUVE scores for the combined PPLM generation sets.

    python src/run_mauve.py --refs val-100len.json --preds pplm-results/all-0.00.json
    python src/run_mauve.py --refs val-100len.json --preds pplm-results/all-0.01.json
    python src/run_mauve.py --refs val-100len.json --preds pplm-results/all-0.04.json
    python src/run_mauve.py --refs val-100len.json --preds pplm-results/all-0.05.json

That's it!

#### Gender Debiasing

To run the debiasing experiment with Backpacks, run

    python src/test_genderbias.py

To run the debiasing experiment with Transformers, run

    python src/test_genderbias.py --use_baseline 1

#### Knowledge Editing Qualitative Example

To run an interactive script in which you can input prefixes like those we used in the paper (e.g., `The MacBook is best known for`), run

    python src/modulate_generate.py

After running an initial script, it will iteratively accept prefixes and generate samples.
You can look at the code to change the affected token.

## Citation
If you use this code or find the ideas useful, please cite the Backpacks paper as well as the FlashAttention paper!

```
@InProceedings{hewitt2023backpack,
  author =      "Hewitt, John and Thickstun, John and Manning, Christopher D. and Liang, Percy",
  title =       "Backpack Language Models",
  booktitle =   "Proceedings of the Association for Computational Linguistics",
  year =        "2023",
  publisher =   "Association for Computational Linguistics",
  location =    "Toronto, Canada",
}

@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
