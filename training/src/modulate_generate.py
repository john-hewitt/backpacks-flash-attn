"""Train a language model from raw samples or argmax."""

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
from models.intervened_models import ReplacedWordLMHeadModel
from demo_generate import generate_iteratively

#from pretrain_from_samples import build
import transformers
from tasks.seq import SequenceLMModel

def print_topk(logits, tokenizer, length, count=10):
  distrib = torch.softmax(logits, dim=-1)
  sorted_distrib, sorted_indices = torch.sort(distrib, descending=True)
  for i in range(count):
    print(tokenizer.decode(sorted_indices[0,length-2,i]), sorted_distrib[0,length-2,i])
  is_prob = distrib[0,length-2,tokenizer(' is')['input_ids'][0]]
  are_prob = distrib[0,length-2,tokenizer(' are')['input_ids'][0]]
  print('~~ are', are_prob.item(), '~~ is', is_prob.item(), '~~ ratio:', is_prob.item() / are_prob.item())


def visualize_word(word, tokenizer, model, count=20, contents=None):
    if contents is None:
      print(word)
      tokens = tokenizer(word)['input_ids'][:1]*512
      print(tokens[0])
      tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')
      #logits1, attns1, contents1 = model({'input_ids': tokens}, return_components=True)
      contents = model.transformer.content_model(tokens) #(bs, nv, s, d)
      contents = contents[0,:,0,:] #(nv, d)

    for i in range(contents.shape[0]):
      print('~~~~~~~~~~~~~~~~~~~~~~~{}~~~~~~~~~~~~~~~~~~~~~~~~'.format(i))
      logits = contents[i,:] @ model.lm_head.weight.t() # (vocab,)
      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      print('~~~Positive~~~')
      for j in range(count):
        print(tokenizer.decode(sorted_indices[j]), '\t',sorted_logits[j].item())
      print('~~~Negative~~~')
      for j in range(count):
        print(tokenizer.decode(sorted_indices[-j-1]), '\t',sorted_logits[-j-1].item())
    return contents
    print()
    print()
    print()

def latex_of_word(word, tokenizer, model, count=20):
  pass

def senses_of_word(word, model):
  tokens = (torch.ones(512)*word).reshape(1,512).long().cuda()
  contents = model.transformer.content_model(tokens) #(bs, nv, s, d)
  contents = contents[0,:,0,:] #(nv, d)
  return contents


def mogrify_word(model, word, out_word, in_word, tokenizer):

  word = tokenizer(word)['input_ids'][0]
  in_word = tokenizer(in_word)['input_ids'][0]
  out_word = tokenizer(out_word)['input_ids'][0]


  def project_out_and_in(
      senses, # (nv, d)
      out_direction, # (d,)
      in_direction, # (d,)
      ):
    #embeddings = embeddings.detach().clone()
    dots = senses @ out_direction / (out_direction  @ out_direction) #(nv)
    normalization = (out_direction @ out_direction) / (in_direction @ in_direction) #(1)
    out_diffs = dots.unsqueeze(1) * out_direction.unsqueeze(0)  #(nv, d)
    in_diffs = dots.unsqueeze(1) * in_direction.unsqueeze(0) * normalization#(nv, d)
    fixed_senses = senses - out_diffs + in_diffs
    #for word in words:
    #  embeddings[word[0]] = fixed_embeddings[word].type(embeddings.type()).detach()
    return fixed_senses


  word_senses = senses_of_word(word, model)
  out_embedding_vector = model.lm_head.weight[out_word]
  in_embedding_vector = model.lm_head.weight[in_word]
  fixed_senses = project_out_and_in(word_senses, out_embedding_vector, in_embedding_vector)
  #visualize_word(None, tokenizer, model, contents=fixed_senses)
  return {word: fixed_senses}




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

@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
def train(config):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-micro-flash-fp16/step_100000.ckpt')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  model = load_non_optimized_model(model.model)
  #model = model.model.to('cuda')

  model.eval()
  for param in model.parameters():
    param.requires_grad = False


  print('Ready!')
  sense_dict = mogrify_word(
      model,
      ' MacBook',
      ' Apple',
      ' HP',
      tokenizer)

  #sense_dict = mogrify_word(
  #    model,
  #    ' Brady',
  #    ' Patriots',
  #    ' Colts',
  #    tokenizer)

  #sense_dict = mogrify_word(
  #    model,
  #    ' Obama',
  #    ' States',
  #    ' Kingdom',
  #    tokenizer)

  #sense_dict.update(mogrify_word(
  #    model,
  #    ' Barack',
  #    ' America',
  #    ' England',
  #    tokenizer))

  # Mogrify
  #word = tokenizer(' Harvard')['input_ids'][0]
  #word_senses = senses_of_word(word, model)
  #new_word_senses = senses_of_word(tokenizer(' Stanford')['input_ids'][0], model)
  #word_senses[12,:] = new_word_senses[12,:]
  #word_senses[7,:] = new_word_senses[7,:]
  #sense_dict = {word: word_senses}

  model = ReplacedWordLMHeadModel(model, sense_dict)
  generate_iteratively(model, tokenizer, inp='I want to eat')


if __name__ == '__main__':
  exp, config = train()
