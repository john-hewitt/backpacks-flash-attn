
from evaluate import load
import click
import json
import random
random.seed(0)

@click.command()
@click.option('--refs', default=None,
    help='path to the config override file')
@click.option('--preds', default=None,
    help='path to the config override file')
def train(refs, preds):
  mauve = load('mauve')

  refs = json.load(open(refs))
  preds = json.load(open(preds))

  if len(preds) > len(refs):
    random.shuffle(preds)
    preds = preds[:len(refs)]

  if len(preds) < len(refs):
    random.shuffle(refs)
    refs = refs [:len(preds)]


  results = mauve.compute(predictions=preds, references=refs, device_id=0)
  print(results)

if __name__ == '__main__':
  exp, config = train()
