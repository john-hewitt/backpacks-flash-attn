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

#from pretrain_from_samples import build
import transformers
from tasks.seq import SequenceLMModel
from utils.generation import sample, greedy_decode


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

def generate_iteratively(model, tokenizer, inp=None):
  if inp is None:
    inp = '<|endoftext|> The capital of France is'
  with torch.autocast(device_type='cuda', dtype=torch.float16):
    while True:
      inp = torch.tensor(tokenizer(inp)['input_ids']).unsqueeze(0).to('cuda')
      #outputs = model.sample(inp, 100)
      #outputs = greedy_decode(inp, model, 50).sequences
      outputs = sample(inp, model, 100).sequences

      print(tokenizer.convert_ids_to_tokens(outputs[0]))
      print(tokenizer.batch_decode(outputs, skip_special_tokens=False))
      inp = input()

@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
def train(config):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-micro-flash-fp16/step_100000.ckpt')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/gpt2s-flash-fp16/last.ckpt')
  model = load_non_optimized_model(model.model)
  #model = model.model.to('cuda')

  model.eval()
  for param in model.parameters():
    param.requires_grad = False
  
  generate_iteratively(model, tokenizer)

if __name__ == '__main__':
  exp, config = train()
