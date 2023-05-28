import click
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
import math
#from data import data
from dataclasses import dataclass
import collections
#from utils import utils
#from utils import config_utils
#from models import models
import torch
from torch import nn

#from pretrain_from_samples import build
import transformers
from tasks.seq import SequenceLMModel
import demo_generate
from models.intervened_models import WeightedBackpackLMHeadModel

def load_non_optimized_model(model):
  model.to('cuda')
  config = model.config
  for k in vars(config):
    if 'fused' in k or 'flash' in k:
      setattr(config, k, False)
  newmodel = type(model)(config)
  newmodel.to('cuda')
  newmodel.load_state_dict(model.state_dict())
  return newmodel

def weights_from_scores(scores, quantile_weights=[1.4,1.2,1.0,0.8]):
  #signs = torch.sign(scores)
  #scores = torch.sqrt(torch.sqrt(torch.sqrt(torch.sqrt(torch.abs(scores)+1e-10))))*signs
  #scores = torch.s
  #scores = torch.sigmoid(scores)#/100)
  max_score = torch.max(scores)
  min_score = torch.min(scores)
  print('q',torch.quantile(scores.reshape(-1), q=torch.tensor([.95, .8, .6]).cuda()))

  quantile_95 = torch.quantile(scores.reshape(-1), q=torch.tensor([.95]).to('cuda'))
  quantile_80 = torch.quantile(scores.reshape(-1), q=torch.tensor([.80]).to('cuda'))
  quantile_60 = torch.quantile(scores.reshape(-1), q=torch.tensor([.60]).to('cuda'))
  #print(quantile_95, quantile_80, quantile_60)

  multiplier = torch.ones_like(scores)
  multiplier = torch.where(quantile_95 < scores, quantile_weights[0], multiplier)
  multiplier = torch.where(torch.logical_and(quantile_80 < scores, scores < quantile_95), quantile_weights[1], multiplier)
  multiplier = torch.where(torch.logical_and(quantile_60 < scores, scores < quantile_80), quantile_weights[2], multiplier)
  multiplier = torch.where(scores < quantile_60, quantile_weights[3], multiplier)
  scores = multiplier
  #print(multiplier)

  #print('q',torch.quantile(scores.reshape(-1), q=torch.tensor([.99, .75, .5, .25, .01]).cuda()))
  print('q',torch.quantile(scores.reshape(-1), q=torch.tensor([.95, .8, .6]).cuda()))
  #scores = (scores - min_score)/(max_score-min_score)
  #print('q1', torch.quantile(scores.reshape(-1), q=torch.tensor([.99, .75, .5, .25, .01]).cuda()))
  #scores = scores*max_upweight + (1-scores)*max_downweight
  #print(torch.sort(scores.reshape(-1)))
  #print('q2',torch.quantile(scores.reshape(-1), q=torch.tensor([.99, .75, .5, .25, .01]).cuda()))
  
  return scores

def non_contextual_localize(target_vector, model, nv, vocsize, tokenizer, verbose=False):
  #plus_stats = torch.zeros(vocsize, nv).cuda()
  scores = torch.zeros(nv, vocsize).cuda()
  for i in tqdm(list(range(vocsize // 512 + 1))):
    tokens = torch.arange(512*i, 512*(i+1))
    tokens = torch.minimum(tokens, torch.tensor(50256))
    tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')

    content = model.transformer.content_model(tokens) # (bs, nv, s, d)
    log_distributions = content @ model.lm_head.weight.t() # (bs, nv, s, voc)

    for seq_index, token_id in enumerate(tokens[0]):
      if token_id == 50256:
        continue
      ld = log_distributions[0,:, seq_index, :] # (nv, vocsize)
      scores[:, token_id] +=  (ld / torch.max(ld,dim=-1, keepdims=True).values) @ target_vector

  sorted_plus, plus_indices = torch.sort(scores.t().reshape(-1), descending=True)
  for i in range(100):
    indx = plus_indices[i]
    score = sorted_plus[i]
    word_indx = indx // scores.shape[0]
    vec_indx = indx % scores.shape[0]
    if verbose:
      print(tokenizer.decode(word_indx), vec_indx, score)
  #for i in range(1,100):
  #  indx = plus_indices[-i]
  #  score = sorted_plus[-i]
  #  word_indx = indx // scores.shape[0]
  #  vec_indx = indx % scores.shape[0]
  #  print(tokenizer.decode(word_indx), vec_indx, score)
  return scores.t()




@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
def train(config):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-micro-flash-fp16/step_100000.ckpt')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  #model.to('cuda')
  model = load_non_optimized_model(model.model)

  model.eval()
  for param in model.parameters():
    param.requires_grad = False


  target_vec = torch.zeros(50264)
  #words = (' hate', ' despise', ' murder', ' terror')
  #words = (' love', ' happy', ' kind', ' generous')
  #words = (' adore',)
  #words = (' aime', ' alors', ' tous', ' les')
  #words = [' nurse', ' medicine', ' doctor', ' health', ' wellness', ' care', ' heal', ' surgery']
  #words += [' Italy', ' Naples', ' Rome', ' pizza', ' wine', ' Medici']
  #words = [' recommendation', ' cars']
  #words = (' crypto', ' Bitcoin', ' Ethereum', ' SBF', ' FTX', ' happy', ' competent', ' responsible')
  #words = (' football', ' soccer', ' basketball', ' game', ' sport', ' baseball', ' hockey', ' field', ' court', ' ball', ' bat', ' goal', ' dunk', ' run', ' race')
  #words = (' football', ' play', ' sport', ' fun', ' soccer', ' ball', ' energetic', ' World', ' Cup')
  #words = (' ',)
  #words = (' basketball', ' football', ' sport', ' hockey)
  #words = (' she',)

  ## Sentiment
  #words = (' love', ' happy', ' kind', ' generous')
  #for word in words:
  #  target_vec[tokenizer(word)['input_ids'][0]] = 1
  #scores = non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
  #scores = weights_from_scores(scores, [2.1,1.8,1.5,1])

  ## Toxicity
  #words = (' hate', ' despise', ' murder', ' terror')
  #for word in words:
  #  target_vec[tokenizer(word)['input_ids'][0]] = 1
  #scores = non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
  #scores = weights_from_scores(scores, [1.4,1.2,1.1,1])

  ## Computer topic
  #words = (' Linux', ' linux', ' computer', ' network', ' compute', ' CPU')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  #for word in words:
  #  target_vec[tokenizer(word)['input_ids'][0]] = 1
  #scores = non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
  #scores = weights_from_scores(scores, [2.1,1.8,1.5,1])

  # Legal
  #words = [' ' + x.strip() for x in open('pplm-data/military.txt')]
  words = [' ' + x.strip() for x in ('sports', )]
  for word in words:
    target_vec[tokenizer(word)['input_ids'][0]] = 1
  scores = non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)

  weight_vec = [3, 3, 2, 1]
  weight_vec = [4, 4, 3, 1]
  weight_vec = [5, 5, 4, 1]
  scores = weights_from_scores(scores, weight_vec)

  ## Music
  #words = (' coffee', ' tea')
  #for word in words:
  #  target_vec[tokenizer(word)['input_ids'][0]] = 1
  #scores = non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
  #scores = weights_from_scores(scores, [2.1,1.8,1.5,1])

  ## Sports topic
  #words = (' football', ' soccer', ' basketball') #, ' game', ' sport', ' baseball', ' hockey', ' field', ' court', ' ball', ' bat', ' goal', ' dunk', ' run', ' race')
  ##words = (' football', ' soccer', ' basketball', ' game', ' sport', ' baseball', ' hockey', ' field', ' court', ' ball', ' bat', ' goal', ' dunk', ' run', ' race')
  #for word in words:
  #  target_vec[tokenizer(word)['input_ids'][0]] = 1
  #scores = non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
  #scores = weights_from_scores(scores, [2.1,1.8,1.5,1])

  #scores = weights_from_scores(scores, [1.4,1.3,1.2,1])
  #scores = weights_from_scores(scores, [1.2,1.2,1.1,1])
  #scores = weights_from_scores(scores, [0.7,0.9,1,1])
  #scores = weights_from_scores(scores, max_upweight=3, max_downweight=1/3)
  #scores = weights_from_scores(scores, max_upweight=1.1, max_downweight=1/1.1)
  model = WeightedBackpackLMHeadModel(model, scores, target_vec, annealing_scale=max(weight_vec)/7.5)
  demo_generate.generate_iteratively(model, tokenizer)

if __name__ == '__main__':
  exp, config = train()
