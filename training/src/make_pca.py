from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from visualize_sim import load_softmax_vecs_for_words
from visualize_sim import load_non_optimized_model
from visualize_sim import load_sense_vecs_for_words

import click
import yaml
import pandas as pd
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import namedtuple
import json
from tqdm import tqdm
import functools
import math
#from data import data
from dataclasses import dataclass
import collections
#from utils import utils
#from utils import config_utils
#from models import models
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr
#from scipy.stats import spearmanr

#from pretrain_from_samples import build
import transformers
from tasks.seq import SequenceLMModel
from utils.generation import sample, greedy_decode
from collections import OrderedDict
#from models.backpack import BackpackLMHeadModel

def get_simverb_words(path, count):
  words = set()
  with open(path) as fin:
    for line in fin:
      line = [x.strip() for x in line.strip().split()]
      words.update({line[0], line[1]})
      if len(words) == count:
        break
  return words


def pca_plot(veclist, pcs=2):
  import numpy as np
  import seaborn as sns
  import matplotlib
  import matplotlib.pyplot as plt

  sns.set_style("whitegrid")
  sem_acc_trf = [0.068, 0.084, 0.239, 0.303, 0.377]
  mauve_trf = [0.95, 0.94, 0.81, 0.62, 0.41]

  SMALL_SIZE = 13
  MEDIUM_SIZE = 14
  BIGGER_SIZE = 15

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


  #font = {'family' : 'normal',
  #        'size'   : 12}
  #
  #matplotlib.rc('font', **font)



  #sem_acc_backpack = [0.074, 0.121, 0.243, 0.353]
  #mauve_backpack = [0.93, 0.91, 0.90, 0.83]
  plt.scatter([x[0] for x in veclist.values()], [x[1] for x in veclist.values()])

  for word in veclist:
    plt.annotate(word,
        xy=veclist[word], xycoords='data',
        #xytext=veclist[word], textcoords='data',
                #arrowprops=dict(arrowstyle="->", color='black', linestyle='--',
                #                connectionstyle="arc3"),
                 c='#593C8F',
                )

  #sns.set_style("whitegrid")
  #sns.boxplot(data=data, palette="deep")
  sns.despine(left=True)

  plt.xlabel('21-Topic Average Control Success')
  plt.ylabel('Overall MAUVE with OpenWebText')
  #plt.plot(mauve_trf, sem_acc_trf)
  #plt.plot(mauve_backpack, sem_acc_backpack)


  #plt.annotate("Unmodified Transformer", (0.068, 0.95))
  sns.despine()

  #plt.annotate("Unmodified Backpack", (0.074, 0.93))

  #plt.annotate("Unmodified Transformer",
  #            xy=(0.068, 0.95), xycoords='data',
  #            xytext=(0.06, 0.65), textcoords='data',
  #            arrowprops=dict(arrowstyle="->", color='black', linestyle='--',
  #                            connectionstyle="arc3"),
  #             c='#593C8F',
  #            )
  #plt.annotate("Unmodified Backpack",
  #            xy=(0.074, 0.93), xycoords='data',
  #            xytext=(0.11, 0.75), textcoords='data',
  #            arrowprops=dict(arrowstyle="->", color='black', linestyle='--',
  #                            connectionstyle="arc3"),
  #            c='#DB5461',
  #            )



              

  plt.title('Topic Control in Generation', fontsize=BIGGER_SIZE)

  #plt.plot(sem_acc_trf, mauve_trf, label='Transformer+PPLM', marker='s',linewidth =2, c='#593C8F')
  #plt.plot(sem_acc_backpack, mauve_backpack, label='Backpack+SenseControl', marker='o', linewidth=2, c='#DB5461')
  plt.legend()
  plt.tight_layout()
  plt.savefig('plt.png')

@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
@click.option('--sense_id', default=12,
    help='Which sense id to use to cluster')
@click.option('--word_count', default=20,
    help='how many words to plot')
def train(config, sense_id, word_count):

  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  model = load_non_optimized_model(model.model)
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', cache_dir='/u/scr/nlp/johnhew/data/huggingface')

  # Get words and filter to those in vocab
  words = get_simverb_words('data/SimVerb-3500.txt', word_count)
  #words = ['Apple', 'iPhone', 'Microsoft', 'Surface', 'Windows', 'iOS', 'iPhone', ]
  words = [tokenizer(' ' + x)['input_ids'] for x in words]
  words = list(filter(lambda x: len(x) == 1, words))
  words = [word[0] for word in words]
  words = [i for i in range(1000)]
  #words = list(filter(lambda x: len(tokenizer(x)['input_ids']) == 1, words))[:word_count]

  # Get vectors for words
  all_senses_dictionary = OrderedDict(load_sense_vecs_for_words(words, tokenizer, model, 'cpu'))
  single_sense_dictionary = OrderedDict({x: v[sense_id,:] for x, v in all_senses_dictionary.items()})
  vec_matrix = torch.stack(list(single_sense_dictionary.values()), dim=0)

  # Perform PCA
  pca_object = PCA(n_components=50)
  transformed_vec_matrix = pca_object.fit_transform(vec_matrix)
  transformed_vec_matrix = TSNE(n_components=2, perplexity=30).fit_transform(transformed_vec_matrix)
  print(transformed_vec_matrix.shape)
  print('explained variance', pca_object.explained_variance_ratio_)
  print('singular values', pca_object.singular_values_)

  # Make a plot
  vec_dictionary = {tokenizer.decode(word): transformed_vec_matrix[i,:] for i, word in enumerate(single_sense_dictionary)}
  print(vec_dictionary)
  pca_plot(vec_dictionary)

if __name__ == '__main__':
  exp, config = train()
