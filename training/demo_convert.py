import torch
import re
from collections import OrderedDict
from src.tasks.seq import SequenceLMModel
from torch.nn import functional as F

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

def remap_state_dict_flash(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r'^transformer.embeddings.position_embeddings.', 'wpe.', key)
    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    state_dict['wpe.weight'] = state_dict['wpe.weight']
    #state_dict['wpe.weight'] = torch.cat((state_dict['wpe.weight'], state_dict['wpe.weight']), dim=0)
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    #state_dict['wte.weight'] = word_embeddings[:50257, :]
    state_dict['wte.weight'] = word_embeddings
    state_dict.pop('lm_head.weight')
    #[:F.pad(
    #    word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
    #)
    #state_dict['transformer.embeddings.word_embeddings.weight'] = state_dict['lm_head.weight'][:config.vocab_size, :]

    # LayerNorm
    ln_weight, ln_bias = state_dict.pop(f'transformer.layers.{config.num_hidden_layers - 1}.norm2.weight'), state_dict.pop(f'transformer.layers.{config.num_hidden_layers - 1}.norm2.bias')
    state_dict['ln_f.weight'] = ln_weight
    state_dict['ln_f.bias'] = ln_bias
    ln_weight, ln_bias = state_dict.pop('transformer.ln_0.weight'), state_dict.pop('transformer.ln_0.bias')
    state_dict['h.0.ln_1.weight'] = ln_weight
    state_dict['h.0.ln_1.bias'] = ln_bias
    for d in range(config.num_hidden_layers):
        ln_weight = state_dict.pop(f'transformer.layers.{d}.norm1.weight')
        ln_bias = state_dict.pop(f'transformer.layers.{d}.norm1.bias')
        state_dict[f'h.{d}.ln_2.weight'] = ln_weight
        state_dict[f'h.{d}.ln_2.bias'] = ln_bias
        if d > 0:
            ln_weight = state_dict.pop(f'transformer.layers.{d - 1}.norm2.weight')
            ln_bias = state_dict.pop(f'transformer.layers.{d - 1}.norm2.bias')
            state_dict[f'h.{d}.ln_1.weight'] = ln_weight
            state_dict[f'h.{d}.ln_1.bias'] = ln_bias

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f'transformer.layers.{d}.mlp.fc1.weight')
        state_dict[f'h.{d}.mlp.c_fc.weight'] = W1.t()
        W2 = state_dict.pop(f'transformer.layers.{d}.mlp.fc2.weight')
        state_dict[f'h.{d}.mlp.c_proj.weight'] = W2.t()
    def key_mapping_mlp(key):
        key = re.sub(r'^transformer.layers.(\d+).mlp.fc1.bias', r'h.\1.mlp.c_fc.bias', key)
        key = re.sub(r'^transformer.layers.(\d+).mlp.fc2.bias', r'h.\1.mlp.c_proj.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        #state_dict.pop(f'h.{d}.attn.bias')  # We don't store this bias
        Wqkv = state_dict.pop(f'transformer.layers.{d}.mixer.Wqkv.weight')
        state_dict[f'h.{d}.attn.c_attn.weight'] = Wqkv.t()
        Wout = state_dict.pop(f'transformer.layers.{d}.mixer.out_proj.weight')
        state_dict[f'h.{d}.attn.c_proj.weight'] = Wout.t()
    def key_mapping_attn(key):
        key = re.sub(r'^transformer.layers.(\d+).mixer.Wqkv.bias', r'h.\1.attn.c_attn.bias', key)
        key = re.sub(r'^transformer.layers.(\d+).mixer.out_proj.bias', r'h.\1.attn.c_proj.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    state_dict = OrderedDict(('transformer.'+k, v) for k, v in state_dict.items())
    #state_dict['lm_head.weight'] = word_embeddings[:50257, :]
    state_dict['lm_head.weight'] = word_embeddings

    return state_dict

def remap_state_dict_gpt2(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r'^wpe.', 'transformer.embeddings.position_embeddings.', key)
    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop('wte.weight')
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(
        word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
    )
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    # LayerNorm
    ln_weight, ln_bias = state_dict.pop('ln_f.weight'), state_dict.pop('ln_f.bias')
    state_dict[f'transformer.layers.{config.num_hidden_layers - 1}.norm2.weight'] = ln_weight
    state_dict[f'transformer.layers.{config.num_hidden_layers - 1}.norm2.bias'] = ln_bias
    ln_weight, ln_bias = state_dict.pop('h.0.ln_1.weight'), state_dict.pop('h.0.ln_1.bias')
    state_dict['transformer.ln_0.weight'] = ln_weight
    state_dict['transformer.ln_0.bias'] = ln_bias
    for d in range(config.num_hidden_layers):
        ln_weight = state_dict.pop(f'h.{d}.ln_2.weight')
        ln_bias = state_dict.pop(f'h.{d}.ln_2.bias')
        state_dict[f'transformer.layers.{d}.norm1.weight'] = ln_weight
        state_dict[f'transformer.layers.{d}.norm1.bias'] = ln_bias
        if d > 0:
            ln_weight = state_dict.pop(f'h.{d}.ln_1.weight')
            ln_bias = state_dict.pop(f'h.{d}.ln_1.bias')
            state_dict[f'transformer.layers.{d - 1}.norm2.weight'] = ln_weight
            state_dict[f'transformer.layers.{d - 1}.norm2.bias'] = ln_bias

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f'h.{d}.mlp.c_fc.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc1.weight'] = W1.t()
        W2 = state_dict.pop(f'h.{d}.mlp.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc2.weight'] = W2.t()
    def key_mapping_mlp(key):
        key = re.sub(r'^h.(\d+).mlp.c_fc.bias', r'transformer.layers.\1.mlp.fc1.bias', key)
        key = re.sub(r'^h.(\d+).mlp.c_proj.bias', r'transformer.layers.\1.mlp.fc2.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        #state_dict[f'h.{d}.attn.bias'] = torch.zeros()  # We don't store this bias
        Wqkv = state_dict.pop(f'h.{d}.attn.c_attn.weight')
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.weight'] = Wqkv.t()
        Wout = state_dict.pop(f'h.{d}.attn.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mixer.out_proj.weight'] = Wout.t()
    def key_mapping_attn(key):
        key = re.sub(r'^h.(\d+).attn.c_attn.bias', r'transformer.layers.\1.mixer.Wqkv.bias', key)
        key = re.sub(r'^h.(\d+).attn.c_proj.bias', r'transformer.layers.\1.mixer.out_proj.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def get_mapping(state_dict_one, state_dict_two):
  """ Returns key mapping of two state dicts """
  mapping = {}
  for key in state_dict_one:
    key_candidate = []
    for key_prime in state_dict_two:
      print(key_prime)
      if (state_dict_one[key].shape == state_dict_two[key_prime].shape and
          torch.all(torch.isclose(state_dict_one[key], state_dict_two[key_prime]))):
        key_candidate.append((key_prime, False))
      if 'out_proj.weight' in key or 'Wqkv.weight' in key or 'mlp.fc1.weight' in key or 'mlp.fc2.weight' in key:
        print(key_prime)
        if (state_dict_one[key].shape and state_dict_two[key_prime].shape and (
            len(state_dict_one[key].shape) == 2 and (len(state_dict_two[key_prime])==2 and(
            (state_dict_one[key].t().shape == state_dict_two[key_prime].shape and
            torch.all(torch.isclose(state_dict_one[key].t(), state_dict_two[key_prime]))))))):
          key_candidate.append((key_prime, True))
    if key_candidate is None:
      raise ValueError("Missing value")
    mapping[key] = key_candidate
  return mapping

def map_state_dict(mapping, state_dict):
  new_state_dict = {}
  for key in state_dict:
    print(key, mapping[key])
    name, transpose = mapping[key][0]
    new_state_dict[name] = state_dict[key].t() if transpose else state_dict[key]
    #new_state_dict[mapping[key][0]] = state_dict[key]
  return new_state_dict

if __name__ == '__main__':
  import transformers
  #model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
  #config = transformers.GPT2Config.from_pretrained('gpt2')
  #print(list(model.transformer.state_dict().keys()))
  #print(list(remap_state_dict_gpt2(model.transformer.state_dict(), config).keys()))
  #for i in list(remap_state_dict_gpt2(model.transformer.state_dict(), config).keys()):
  #  print(i)
  #mapping = get_mapping(
  #  remap_state_dict_gpt2(model.transformer.state_dict(), config),
  #  model.transformer.state_dict()
  #  )
  #print(mapping)
  tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
  model_flash = SequenceLMModel.load_from_checkpoint('checkpoints/gpt2s-flash-fp16/last.ckpt')
  model_flash = load_non_optimized_model(model_flash.model)
  config = model_flash.config
  model = transformers.GPT2LMHeadModel(config)
  new_state_dict = remap_state_dict_flash(model_flash.state_dict(), config)
  for key in model.state_dict():
    if '.attn.masked_bias' in key or '.attn.bias' in key:
      print(key)
      new_state_dict[key] = model.state_dict()[key]

  model.load_state_dict(new_state_dict)
  model.to('cuda')
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
      config.pad_token_id = config.eos_token_id
  print(tokenizer.batch_decode(
    model.generate(
      torch.tensor(tokenizer('<|endoftext|>The capital of the United States is Washington D.C. The capital of England is')['input_ids']).unsqueeze(0).to('cuda'),
      pad_token_id=50256,
      do_sample=True,
      #eos_token=50256,
      max_length = 150
      )))

