"""Train a language model from raw samples or argmax."""

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
#from models.backpack import BackpackLMHeadModel

def load_non_optimized_model(model):
  """
  Reloads model with all fused CUDA kernel stuff
  turned off, so it can be run on old GPUs.
  """
  model.to('cuda')
  config = model.config
  for k in vars(config):
    if 'fused' in k or 'flash' in k:
      setattr(config, k, False)
  newmodel = type(model)(config)
  newmodel.to('cuda')
  newmodel.load_state_dict(model.state_dict())
  return newmodel


def load_softmax_vecs_for_words(wordlist, tokenizer, model, final_device):
  """
  Loads the softmax matrix embedding for each of
  a list of words. (This is tied with the input embedding
  in GPT-2 models.)
  Assumes the wordlist has been filtered to ensure
  that it's all words that occur in the vocab.

  Returns:
    Dictionary of {word string: Torch tensor}
  """
  vec_dictionary = {}
  for word in tqdm(wordlist):
    #spaced_word = mogrify_word(word)
    lm_head = getattr(model, 'lm_head', None)
    if lm_head is None:
      lm_head = getattr(model, 'lm_loss', None)
    #idx = tokenizer(spaced_word)['input_ids'][0] # assumes 1
    #vec_dictionary[word] = lm_head.weight[idx,:].detach().to(final_device)
    #idx = tokenizer(spaced_word)['input_ids'] # assumes 1
    idx = word
    vec_dictionary[word] = lm_head.weight[idx,:].detach().to(final_device)
  return vec_dictionary


def load_sense_vecs_for_words(wordlist, tokenizer, model, final_device):
  """
  Loads sense vectors per a list of words.
  Model must be a backpack network.
  Assumes the wordlist has been filtered to ensure
  that it's all words that occur in the vocab.

  Returns:
    Dictionary of {word string: Torch tensor}
  """
  vec_dictionary = {}
  for word in tqdm(wordlist):
    #spaced_word = mogrify_word(word)
    #idx = tokenizer(spaced_word)['input_ids'][0] # assumes 1
    #tokens = torch.tensor(idx).reshape(1,1).to('cuda')
    #vecs = model.transformer.content_model(tokens) # (1, nv, 1, ndim)
    #vec_dictionary[word] = vecs[0,:,0,:].detach().to(final_device) # (nv, ndim)
    #idx = tokenizer(spaced_word)['input_ids']
    idx = word
    tokens = torch.tensor(idx).reshape(1,-1).to('cuda')
    vecs = model.transformer.content_model(tokens) # (1, nv, seqlen, ndim)
    vec_dictionary[word] = torch.mean(vecs[0,:,:,:],dim=1).detach().to(final_device) # (nv, ndim)
  return vec_dictionary


def get_similarity_fns(multivec_methods, model):
  """
  From a model definition, returns a list of similarity functions to try
  """
  metrics = {}
  def flat_cosine(vec1, vec2):
    vec1 = vec1.reshape(-1)
    vec2 = vec2.reshape(-1)
    dot =  vec1.dot(vec2)
    norm_prod = vec1.norm(p=2) * vec2.norm(p=2)
    return dot/norm_prod
  metrics['Cos'] = flat_cosine

  def _get_all_cosines(vec1, vec2):
    dot =  vec1 @ vec2.t() # (nv, nv)
    vec1_norms = vec1.norm(p=2,dim=-1) # (nv, )
    vec2_norms = vec2.norm(p=2,dim=-1) # (nv, )
    norm_prods = torch.outer(vec1_norms, vec2_norms)
    cosines = dot / norm_prods
    return cosines

  def min_pairwise_cosines(vec1, vec2):
    all_cosines = _get_all_cosines(vec1, vec2)
    return torch.min(torch.diagonal(all_cosines))

  def max_pairwise_cosines(vec1, vec2):
    all_cosines = _get_all_cosines(vec1, vec2)
    return torch.max(torch.diagonal(all_cosines))

  def min_all_cosines(vec1, vec2):
    all_cosines = _get_all_cosines(vec1, vec2)
    return torch.min(all_cosines)

  def max_all_cosines(vec1, vec2):
    all_cosines = _get_all_cosines(vec1, vec2)
    return torch.max(all_cosines)

  def one_matched_cosine(vec1, vec2, i):
    vec1 = vec1[i, :]
    vec2 = vec2[i, :]
    return flat_cosine(vec1, vec2)

  if multivec_methods:
    metrics['MinPair'] = min_pairwise_cosines
    metrics['MaxPair'] = max_pairwise_cosines
    metrics['MinAll'] = min_all_cosines
    metrics['MaxAll'] = max_all_cosines
    for i in range(model.config.num_content_vectors):
      metrics['CosSense{}'.format(i)] = functools.partial(
          one_matched_cosine,
          i=i
          )
  return metrics

@click.command()
@click.option('--checkpoint', default=None,
    help='path to the checkpoint to load')
@click.option('--multivec_methods', default=False,
    help='use similarity methods that make use of multiple vecs per word')
@click.option('--use_softmax', default=True,
    help='use the softmax embeddings')
@click.option('--name', default=None,
    help='name this eval')
def train(checkpoint, multivec_methods, use_softmax, name):
  print('Loading tokenizer and model')

  if checkpoint in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'EleutherAI/gpt-j-6B', 'xlnet-base-cased', 'xlnet-large-cased'}:
    model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir='/u/scr/nlp/johnhew/data/huggingface')
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, cache_dir='/u/scr/nlp/johnhew/data/huggingface')
  else:
    model = SequenceLMModel.load_from_checkpoint(checkpoint)
    model = load_non_optimized_model(model.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', cache_dir='/u/scr/nlp/johnhew/data/huggingface')

  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  # Get the dataset
  #print('Loading data')
  #datasets = {
  #    'simlex999': load_simlex999('data/SimLex-999/SimLex-999.txt'),
  #    'simverb3500': load_simverb3500('data/SimVerb-3500.txt'),
  #    'rg65': load_rg65('data/RG65.csv'),
  #    'ws353': load_ws353('data/WS353.csv'),
  #    }
  #for dataset_name, dataset in datasets.items():
  #  print('Dataset {} has {} examples'.format(dataset_name, len(dataset)))

  # filter to words in tokenizer
  #print('Filtering data to tokenizer')
  #examples = filter_to_tokenizer(examples, tokenizer)
  #print('There are {} examples after filtering'.format(len(examples)))

  # Define the word set
  #words = set([x.word1 for x in itertools.chain(*datasets.values())]).union(
  #    [x.word2 for x in itertools.chain(*datasets.values())])
  words = range(50256)
  #words = set([x.word1 for x in examples] + [x.word2 for x in examples])

  # Define the similarity fns: 
  print('Defining similarity fns')
  sim_fns = get_similarity_fns(multivec_methods, model)

  # Define the vector dictionary
  print('Loading/generating vectors')
  if use_softmax:
    print('Using softmax vectors')
    vec_dictionary = load_softmax_vecs_for_words(words, tokenizer, model, 'cuda')
  else:
    print('Using sense vectors')
    vec_dictionary = load_sense_vecs_for_words(words, tokenizer, model, 'cuda')


  sim_fn = sim_fns['MinPair']
  #sim_fn = sim_fns['MinPair']
  #sim_fn = sim_fns['Cos']
  while True:
    test_word = tokenizer(input().strip('\n'))['input_ids'][0]
    similarities = {word: sim_fn(vec_dictionary[test_word], vec_dictionary[word]) for word in tqdm(vec_dictionary)}
    for i, (word, sim) in enumerate(sorted(similarities.items(), key=lambda x:-similarities[x[0]])):
      if i < 100: # or i > len(similarities) - 20:
        print(tokenizer.decode(word) + '\t' + tokenizer.decode(test_word) + '\t' + str(sim))

  # evaluate the similarity fns
  #evaluate_fns(sim_fns, examples, vec_dictionary)
  #eval_dict = {}
  #for dataset_name, data in datasets.items():
  #  eval_dict[dataset_name] = {}
  #  for sim_fn_name, sim_fn in sim_fns.items():
  #    print('Evaluating on {} with sim fn {}'.format(dataset_name, sim_fn_name))
  #    sims = get_sims_for_fn(sim_fn, data, vec_dictionary)
  #    evals = get_evals_of_sims(sims, data)
  #    eval_dict[dataset_name][sim_fn_name] = evals
  #prettify_results(eval_dict, name)
  #print(json.dumps(eval_dict, sort_keys=True, indent=4))



if __name__ == '__main__':
  exp, config = train()
