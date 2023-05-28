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

def print_topk(logits, tokenizer, length, count=10):
  distrib = torch.softmax(logits, dim=-1)
  sorted_distrib, sorted_indices = torch.sort(distrib, descending=True)
  d = {}
  for i in range(count):
    print(tokenizer.decode(sorted_indices[0,length-2,i]), sorted_distrib[0,length-2,i])
    d[tokenizer.decode(sorted_indices[0,length-2,i])] = sorted_distrib[0,length-2,i].item()
  is_prob = distrib[0,length-2,tokenizer(' he')['input_ids'][0]]
  are_prob = distrib[0,length-2,tokenizer(' she')['input_ids'][0]]
  print('~~ she', are_prob.item(), '~~ he', is_prob.item(), '~~ ratio:', is_prob.item() / are_prob.item())
  is_prob = distrib[0,length-2,tokenizer(' his')['input_ids'][0]]
  are_prob = distrib[0,length-2,tokenizer(' her')['input_ids'][0]]
  print('~~ her', are_prob.item(), '~~ his', is_prob.item(), '~~ ratio:', is_prob.item() / are_prob.item())
  is_prob = distrib[0,length-2,tokenizer(' John')['input_ids'][0]]
  are_prob = distrib[0,length-2,tokenizer(' Mary')['input_ids'][0]]
  print('~~ Mary', are_prob.item(), '~~ John', is_prob.item(), '~~ ratio:', is_prob.item() / are_prob.item())
  print(d)

def compute_counterfactual(tokens, content, contextualization, word_id, vector_index, percent):
  contextualization = contextualization.clone()
  #contextualization[0, vector_index, :, :] *= torch.tensor(percent).to(contextualization.device)
  target_word_indices = ((tokens == word_id).nonzero(as_tuple=True))
  outputs = contextualization @ content
  for index in target_word_indices[1]:
    print(index)
    contextualization[0, vector_index, :, index] *= percent 
  outputs = torch.sum(contextualization @ content, dim=1) # (bs, s, d)
  return outputs

def modulate(model, context, word_id, vector_index, tokenizer):
  print(tokenizer.convert_ids_to_tokens(tokenizer(context)['input_ids']))
  tokens = tokenizer(context)['input_ids']
  length = len(tokens)
  tokens = tokens + tokenizer('<|endoftext|>')['input_ids']*(512-len(tokens))
  tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')

  # create the outputs
  with torch.autocast(device_type='cuda', dtype=torch.float16):
    content = model.model.transformer.content_model(tokens) # (bs, nv, s, d)
    _context_hiddens = model.model.transformer.gpt2_model(tokens) # (bs, nv, s, s)
    contextualization = model.model.transformer.contextualization_attn(_context_hiddens)

    while True:
      percent = float(input())
      output = compute_counterfactual(tokens, content, contextualization, word_id, vector_index, percent)
      logits = model.model.lm_head(output)
      print_topk(logits, tokenizer, length)


def modulate(model, context, word_id, vector_index, tokenizer):
  while True:
    context = (input('Context!'))
    print(tokenizer.convert_ids_to_tokens(tokenizer(context)['input_ids']))
    tokens = tokenizer(context)['input_ids']
    length = len(tokens)
    tokens = tokens + tokenizer('<|endoftext|>')['input_ids']*(512-len(tokens))
    tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')

    # create the outputs
    with torch.autocast(device_type='cuda', dtype=torch.float16):
      content = model.model.transformer.content_model(tokens) # (bs, nv, s, d)
      _context_hiddens = model.model.transformer.gpt2_model(tokens) # (bs, nv, s, s)
      contextualization = model.model.transformer.contextualization_attn(_context_hiddens)

      percent = float(input('Percent!'))
      output = compute_counterfactual(tokens, content, contextualization, word_id, vector_index, percent)
      logits = model.model.lm_head(output)
      print_topk(logits, tokenizer, length)

@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
def train(config):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-micro-flash-fp16/step_100000.ckpt')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  model.to('cuda')

  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  #token_id = tokenizer(' it')['input_ids'][0]
  token_id = tokenizer(' heart')['input_ids'][0]
  token_id = tokenizer(' nurse')['input_ids'][0]
  #token_id = tokenizer(' driver')['input_ids'][0]
  context = 'The nurse came into the room, and X'
  context = 'My CEO said X'
  context = 'The developer said that X'
  #context = 'Registered nurses (RNs) provide and coordinate patient care, educate patients and the public about various health conditions, and provide advice and emotional support to patients and their families. My nurse said X'
  #context = 'The nurse went into the room to talk to the patient. After looking at the meds and adjusting the IV, X'
  #context = "I really hate my chair because it is X"
  #context = "I feel in my heart that love is a X"
  #context = '"Hi, I\'ll be your nurse; my name is X'
  #context = 'The truck driver walked to the counter to pick up X'
  print(token_id)
  print(tokenizer.convert_ids_to_tokens(token_id))

  #modulate(model, context, token_id, 5, tokenizer)
  modulate(model, context, token_id, 10, tokenizer)
  #modulate(model, context, token_id, 13, tokenizer)

if __name__ == '__main__':
  exp, config = train()
