import click
import yaml
import math
import random
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
import itertools
from models.intervened_models import WeightedBackpackLMHeadModel
import test_sentiment
import rank_vocab
import evaluate
import stanza
from scipy.special import expit

#toxicity = evaluate.load("toxicity", module_type="measurement")
#model_path = "cardiffnlp/tweet-topic-21-multi"
#sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
#sentiment_task.model.to('cuda')
#sentiment_task.device = sentiment_task.model.device

model_path = f"cardiffnlp/tweet-topic-21-multi"
topic_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
model.to('cuda')
model.eval()
for i in model.parameters():
  i.requires_grad = False
class_mapping = model.config.id2label

nlp = stanza.Pipeline('en', processors='tokenize')


gpt_tokenier = transformers.GPT2Tokenizer.from_pretrained('gpt2-large')
gpt_model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-large').to('cuda')

gpt_model.to('cuda')
gpt_model.eval()
for i in gpt_model.parameters():
  i.requires_grad = False


def topic_scores(text):
  tokens = topic_tokenizer(text, return_tensors='pt').to('cuda')
  output = model(**tokens)
  scores = output[0][0].cpu().detach().numpy()
  scores = expit(scores)
  return {class_mapping[i]: float(scores[i]) for i in range(len(scores))}


def count_coverage(generations, dictionary):
  total = 0
  contains = 0
  all_words = set()
  for generation in generations:
    words = nlp(generation)
    words = list(itertools.chain(*[[y.text for y in x.words] for x in nlp(generation).sentences]))
    all_words.update(words)
    contains += len(list(filter(lambda x:x, [x in dictionary for x in words])))
    total += len(words)
  return contains/total, len(all_words)


def gpt2medium_likelihood(generations):
  nlls = []
  for generation in generations:
    tokens = gpt_tokenier(generation, return_tensors='pt').to('cuda')
    outputs = gpt_model(**tokens, labels=tokens['input_ids'].clone())
    nlls.append(outputs.loss.detach().cpu().item())
  return sum(nlls)/len(nlls), sum([x<1 for x in nlls])/len(nlls)



@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
@click.option('--strength', default=0,
    help='path to the config override file')
@click.option('--words_path', default='topic_classes/sports.txt',
    help='path to the one-word-per-line list')
@click.option('--generations_path', default=None,
    help='path to already-generated text if it exists')
@click.option('--seed', default=83570,
    help='path to already-generated text if it exists')
def train(config, strength, words_path, generations_path, seed):

  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  model = rank_vocab.load_non_optimized_model(model.model)

  model.eval()
  for param in model.parameters():
    param.requires_grad = False


  name_dict = [ 'sports', 'diaries_&_daily_life', 'fitness_&_health', 'family', 'youth_&_student_life', 'gaming', 'relationships', 'news_&_social_concern', 'learning_&_educational', 'celebrity_&_pop_culture', 'travel_&_adventure', 'food_&_dining', 'business_&_entrepreneurs', 'science_&_technology', 'fashion_&_style', 'music', 'other_hobbies', 'film_tv_&_video', 'arts_&_culture']
  name_dict = {x.replace('&_', ''):x for x in name_dict}

  reverse_name_dict = {v:k for k, v in name_dict.items()}
  if generations_path:
    words_path = 'topic_classes/{}.txt'.format(os.path.basename(generations_path).split('-')[0])

  target_vec = torch.zeros(50264)
  words = [' ' + x.strip() for x in open(words_path)]
  for word in words:
    target_vec[tokenizer(word)['input_ids'][0]] = 1

  pplm_dict = {
		'arts_culture': 'arts_&_culture', 
		'business_entrepreneurs':  'business_&_entrepreneurs',
		'celebrity_pop_culture':  'celebrity_&_pop_culture',
		'diaries_daily_life': 'diaries_&_daily_life',
		'family': 'family',
		'fashion_style': 'fashion_&_style',
		'film_tv_video': 'film_tv_&_video',
		'fitness_health': 'fitness_&_health',
		'food_dining': 'food_&_dining',
		'gaming': 'gaming',
		'music': 'music',
		'news_social_concern': 'news_&_social_concern',
		'other_hobbies': 'other_hobbies',
		'relationships': 'relationships',
		'sports': 'sports',
		'travel_adventure': 'travel_&_adventure',
		'youth_student_life': 'youth_&_student_life'
		}


  modifier_dict = {
      0: [1,1,1,1],
      1: [1.5,1.5,1.3,1],
      2: [2.2,2.2,1.5,1],
      3: [3.3,3.3,3,1],
      #5: [4.5,4.5,3.5,1],
      }

  modifiers = modifier_dict[strength]


  if not generations_path:
    scores = rank_vocab.non_contextual_localize(target_vec.to('cuda'), model, 16, 50264, tokenizer)
    scores = rank_vocab.weights_from_scores(scores, modifiers)
    model = WeightedBackpackLMHeadModel(model, scores, target_vec, annealing_scale=max(modifiers)/7.5)
    generations = test_sentiment.generate_k(model, tokenizer, 500, batch=8)
  else:
    generations = json.load(open(generations_path))
  topics = [topic_scores(x) for x in tqdm(generations, desc='topic')]
  path_topic = os.path.basename(words_path).split('.')[0]
  correct_topic = name_dict[path_topic] if generations_path is None else pplm_dict[os.path.basename(generations_path).split('-')[0]]
  both = [{
    'generations': gen,
    'topics': top,
    'correct_topic': correct_topic,
    'above_50': top[correct_topic]>0.5 ,
    'above_30': top[correct_topic]>0.3 }
    for gen, top in zip(generations, topics)
    ]
  success = sum([x['above_50'] for x in both])/len(both)
  above_30= sum([x['above_30'] for x in both])/len(both)
  percent_dict, all_words = count_coverage(generations, [x.strip() for x in words])
  avg_likelihood, repetition_count = gpt2medium_likelihood(generations)
  print('success', success)
  print('above_30', above_30)
  print('percent_in_seed_dict', percent_dict)
  print('avg_likelihood', avg_likelihood)
  print('word_count', all_words)
  print('repetition_count', repetition_count)
  if not generations_path:
    path = 'logs/backpack-results_{}_{}__topic_summary'.format(os.path.basename(words_path).split('.')[0], strength)
    with open('backpack-topic-results/{}-results-{}'.format(os.path.basename(words_path).split('.')[0], strength), 'w') as fout:
      json.dump(both, fout)
  else:
    path = 'logs/pplm-results_{}-topic_summary'.format(os.path.basename(generations_path))
    with open(path+'.results_generations', 'w') as fout:
      json.dump(both, fout)
  with open(path, 'w') as fout:
    fout.write("Semantic_success {}\n".format(success))
    fout.write("Percent_in_seed_dictionary {}\n".format(percent_dict))
    fout.write("GPT2Medium_avg_log_likelihood {}\n".format(avg_likelihood))
    fout.write("above_30 {}\n".format(above_30))
    fout.write("word_count {}\n".format(all_words))
    fout.write("repetition_count {}\n".format(repetition_count))

if __name__ == '__main__':
  exp, config = train()
