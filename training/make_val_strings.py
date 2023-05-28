import numpy
import transformers
import json

a=numpy.load('/u/scr/nlp/johnhew/data/openwebtext/cache/tokenizer_name-gpt2-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/validation.npy')
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

res = []
for elt in tokenizer.decode(a).split('<|endoftext|>'):
  if not elt:
    continue
  res.append(tokenizer.decode(tokenizer('<|endoftext|>' + elt)['input_ids'][:100]))
json.dump(res, open('val-100len.json', 'w'))

