"""Train a language model from raw samples or argmax."""

import click
import yaml
import math
import re
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

import re

def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
				from https://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates
    """
    conv = {
        #'&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)

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
        print(tokenizer.decode(sorted_indices[j]), '\t','{:.2f}'.format(sorted_logits[j].item()))
      print('~~~Negative~~~')
      for j in range(count):
        print(tokenizer.decode(sorted_indices[-j-1]), '\t','{:.2f}'.format(sorted_logits[-j-1].item()))
    return contents
    print()
    print()
    print()

def latex_of_word(word, tokenizer, model, count=20, contents=None):
  if contents is None:
    print(word)
    tokens = tokenizer(word)['input_ids'][:1]*512
    print(tokens[0])
    tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')
    #logits1, attns1, contents1 = model({'input_ids': tokens}, return_components=True)
    contents = model.transformer.content_model(tokens) #(bs, nv, s, d)
    contents = contents[0,:,0,:] #(nv, d)

  positive_word_strings = []
  negative_word_strings = []
  for i in range(contents.shape[0]):
    positive_sense_strings = []
    negative_sense_strings = []
    logits = contents[i,:] @ model.lm_head.weight.t() # (vocab,)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    for j in range(count):
     positive_sense_strings.append(tokenizer.decode(sorted_indices[j]))#, '\t','{:.2f}'.format(sorted_logits[j].item()))
    for j in range(count):
      negative_sense_strings.append(tokenizer.decode(sorted_indices[-j-1]))#, '\t','{:.2f}'.format(sorted_logits[-j-1].item()))
    positive_word_strings.append(positive_sense_strings)
    negative_word_strings.append(negative_sense_strings)

  print('Positive')
  string = '&\t'.join(str(j) for j in range(contents.shape[0]))
  print(tex_escape(string)+'\\\\')
  for i in range(count):
    string = '&\t'.join(positive_word_strings[j][i] for j in range(contents.shape[0]))
    print(tex_escape(string)+'\\\\')

  print('Negative')
  string = '&\t'.join(str(j) for j in range(contents.shape[0]))
  print(tex_escape(string)+'\\\\')
  for i in range(count):
    string = '&\t'.join(negative_word_strings[j][i] for j in range(contents.shape[0]))
    print(tex_escape(string)+'\\\\')
  print()
  print()
  return contents


def mogrify_word(model, word, out_word, in_word, tokenizer):

  word = tokenizer(word)['input_ids'][0]
  in_word = tokenizer(in_word)['input_ids'][0]
  out_word = tokenizer(out_word)['input_ids'][0]

  def senses_of_word(word):
    print(word)
    tokens = (torch.ones(512)*word).reshape(1,512).long().cuda()
    #tokens = tokenizer(word)['input_ids'][:1]*512
    print(tokens[0])
    #tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')
    contents = model.transformer.content_model(tokens) #(bs, nv, s, d)
    contents = contents[0,:,0,:] #(nv, d)
    return contents

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


  word_senses = senses_of_word(word)
  out_embedding_vector = model.lm_head.weight[out_word]
  in_embedding_vector = model.lm_head.weight[in_word]
  fixed_senses = project_out_and_in(word_senses, out_embedding_vector, in_embedding_vector)
  visualize_word(None, tokenizer, model, contents=fixed_senses)




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
@click.option('--word', default=None,
    help='word to use and exit')
def train(config, word):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-micro-flash-fp16/step_100000.ckpt')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  model = load_non_optimized_model(model.model)
  #model = model.model.to('cuda')

  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  #_avg = visualize_word(' it', tokenizer, model)
  #man_avg = visualize_word(' hate', tokenizer, model)
  #king_avg = visualize_word(' dislike', tokenizer, model)
  #queen_avg = visualize_word('\'s', tokenizer, model)
  #woman_avg = visualize_word('.', tokenizer, model)
  #woman_avg = visualize_word(' mix', tokenizer, model,
  #    contents=king_avg - man_avg + woman_avg)

  #man_avg = visualize_word(  ' man', tokenizer, model)
  #king_avg = visualize_word( ' king', tokenizer, model)
  #queen_avg = visualize_word(' queen', tokenizer, model)
  #woman_avg = visualize_word(' woman', tokenizer, model)
  #woman_avg = visualize_word(' mix', tokenizer, model,
  #    contents=(king_avg + man_avg + woman_avg + queen_avg)/4)

  #man_avg = visualize_word(  ' nurse', tokenizer, model)
  #king_avg = visualize_word( ' doctor', tokenizer, model)
  #queen_avg = visualize_word(' bank', tokenizer, model)
  #woman_avg = visualize_word(' hate', tokenizer, model)
  #woman_avg = visualize_word(' breakfast', tokenizer, model)
  #woman_avg = visualize_word(' ', tokenizer, model,
  #    contents=(king_avg + man_avg + woman_avg + queen_avg)/4)

  def word_arithmetic(word_string):
    print(word_string)
    components =re.split(r'\+|-', word_string)[1:]
    print(components)
    all_contents = None
    pluses_minuses = list(filter(lambda x: x in {'+','-'}, word_string))
    for word, pm in zip(components, pluses_minuses):
      print(pm, word)
      sign = 1 if pm == '+' else -1
      tokens = (torch.ones(512)*tokenizer(word)['input_ids'][0]).reshape(1,512).long().cuda()
      contents = model.transformer.content_model(tokens) #(bs, nv, s, d)
      if all_contents is None:
        all_contents = sign*contents[0,:,0,:] #(nv, d)
      else:
        all_contents += sign*contents[0,:,0,:] #(nv, d)
    return all_contents


  if word:
    contents = None
    if '+' in word or '-' in word:
      contents = word_arithmetic(word)
    latex_of_word(word, tokenizer, model, count=10, contents=contents)
    return

  print('Ready!')
  while True:
    _ = visualize_word(  input().strip('\n'), tokenizer, model, count=20)
    #_ = mogrify_word(model, tokenizer(' Harvard')['input_ids'][0], tokenizer(' Massachusetts')['input_ids'][0], tokenizer(input().strip('\n'))['input_ids'][0], tokenizer)
   # _ = mogrify_word(
   #     model,
   #     ' MacBook',
   #     ' Apple',
   #     input(),
   #     tokenizer)

#    _ = mogrify_word(
#        model,
#        ' Harvard',
#        ' Massachusetts',
#        input(),
#        tokenizer)
#
if __name__ == '__main__':
  exp, config = train()
