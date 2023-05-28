import click
import yaml
import math
from collections import Counter
from transformers import pipeline
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
from utils.generation import sample, greedy_decode
import transformers
from tasks.seq import SequenceLMModel
import demo_generate
from models.intervened_models import WeightedBackpackLMHeadModel
import rank_vocab
import evaluate
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task.model.to('cuda')
sentiment_task.device = sentiment_task.model.device


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


def generate_k(model, tokenizer, total_strings, prefix='<|endoftext|>', batch=256, length=100):
  with torch.autocast(device_type='cuda', dtype=torch.float16):
    strings = []
    for i in tqdm(range(total_strings//batch + 1)):
      inp = torch.tensor(tokenizer(prefix)['input_ids']).unsqueeze(0).to('cuda').expand(batch, -1)
      outputs = sample(inp, model, length).sequences
      strings.extend(tokenizer.batch_decode(outputs, skip_special_tokens=False))
      if len(strings) >= total_strings:
        return strings[:total_strings]


@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
@click.option('--sentiment_modifier', default=0,
    help='path to the config override file')
def train(config, sentiment_modifier):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  #model.to('cuda')
  model = load_non_optimized_model(model.model)

  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  # Sentiment
  target_vec = torch.zeros(50264)
  words = (' happy', ' love', ' kind', ' generous',)
  for word in words:
    target_vec[tokenizer(word)['input_ids'][0]] = 1
  scores = rank_vocab.non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
  #scores = rank_vocab.weights_from_scores(scores, [1.4,1.2,1.1,1])
  if sentiment_modifier == 0:
    print('SentMod', sentiment_modifier)
    modifiers = [1,1,1,1]
    scores = rank_vocab.weights_from_scores(scores, modifiers)
  elif sentiment_modifier == 1:
    print('SentMod', sentiment_modifier)
    modifiers = [1.6,1.4,1.2,1]
    scores = rank_vocab.weights_from_scores(scores, modifiers)
  elif sentiment_modifier == 2:
    print('SentMod', sentiment_modifier)
    modifiers = [2.4,1.8,1.3,1]
    scores = rank_vocab.weights_from_scores(scores, modifiers)
  #elif sentiment_modifier == -1:
  #  print('SentMod', sentiment_modifier)
  #  target_vec = torch.zeros(50264)
  #  for word in (' sad', ' unhappy', ' dislike'):
  #    target_vec[tokenizer(word)['input_ids'][0]] = 1
  #  scores = rank_vocab.non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
  #  scores = rank_vocab.weights_from_scores(scores, [1.6,1.4,1.2,1])

  model = WeightedBackpackLMHeadModel(model, scores, target_vec, annealing_scale=max(modifiers)/12.5)
  generations = generate_k(model, tokenizer, 10000, batch=36)
  print(generations)
  results = [sentiment_task(x) for x in generations]
  print(results)
  counter = Counter([x[0]['label'] for x in results])
  print(counter)
  #print(Counter([x['label'] for x in counter]))
  print({x: counter[x]/len(results) for x in counter})
  with open('results/sentiment-{}-2.txt'.format(sentiment_modifier), 'w') as fout:
    json.dump(generations, fout)

if __name__ == '__main__':
  exp, config = train()
