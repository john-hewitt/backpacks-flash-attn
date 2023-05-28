from torch import nn
import math
import torch
from collections import namedtuple
from utils.generation import GenerationMixin



def create_content_soft_mask(
    content_weights, # (vocsize, nv)
    input_ids, # (bs, seqlen)
    scores, # (bs, seqlen, nv)
    ):
  vocsize, nv = content_weights.shape
  bs, seqlen = input_ids.shape
  input_ids = input_ids.unsqueeze(2).expand(*input_ids.shape, nv) # (bs, seqlen, nv)
  content_weights = content_weights.unsqueeze(0).expand(bs, vocsize, nv) #(bs, vocsize, nv)
  weights = torch.gather(content_weights, dim=1, index=input_ids) #(bs, seqlen, nv)
  weights = weights*scores + torch.ones_like(weights).to(weights.device)*(1-scores)
  return weights


def get_sense_vector_of_word(word_id, model, sense_index):
  inputs = (torch.ones(1, model.config.n_positions).to(word_id.device)*word_id).long()
  senses = model.transformer.content_model(inputs) # (bs, nv, s, d)
  return senses[0,sense_index,0,:] #(d,)


def mask_annealing(model, input_ids, target_vector, content, annealing_scale=0.1, upweight_nearby=True):

  bs, seqlen = input_ids.shape
  vocsize = target_vector.shape[0]

  # Non-negative vocab log-probs
  content = torch.nn.ReLU()(content @ model.lm_head.weight.t()) #(bs, nv, s, voc)
  input_ids = input_ids.reshape(bs, 1, 1, seqlen) #(bs, 1, 1, seqlen)
  input_ids = input_ids.expand(-1, 16, seqlen, -1) #(bs, 16, 1, seqlen)

  # Words-senses non-neg log-probs
  sims = torch.gather(content, dim=3, index=input_ids) #(bs, nv, seq, seq)
  sims = torch.where(sims>0.0, sims, 0)

  # Overall sense satisfaction score
  sims = torch.sum(sims,dim=3) #(bs, nv, seq)

  # Squash 'em
  scores = torch.sigmoid(-annealing_scale*sims + 6) #(bs, nv, seq)

  # Stuff closer to the most recent tokens generated should be upweighted
  if upweight_nearby:
    score_modifier = (1+torch.arange(seqlen)/100).reshape(1,1,seqlen).to(scores.device)
    scores *= score_modifier
  return scores




class WeightedBackpackLMHeadModel(nn.Module, GenerationMixin):

  def __init__(self, backpack_network, content_weights, target_weight, annealing_scale, anneal=True, upweight_nearby=True):
    super().__init__()
    self.backpack_network = backpack_network
    self.content_weights = content_weights # (nv, vocsize)
    self.target_weight = target_weight.to('cuda')
    self.annealing_scale = annealing_scale
    self.anneal = anneal
    self.upweight_nearby = upweight_nearby


  def forward(self, input_ids, position_ids=None, inference_params=None):
    """
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
    """

    # unchanged
    contextl_hidden_states = self.backpack_network.transformer.gpt2_model(input_ids, position_ids=position_ids, inference_params=inference_params)
    contextualization = self.backpack_network.transformer.contextualization_attn(contextl_hidden_states) # (bs, nv, s, s)

    # Compute content and weight
    content = self.backpack_network.transformer.content_model(input_ids, position_ids, inference_params) # (bs, nv, s, d)
    if self.anneal:
      annealing_scores = mask_annealing(self.backpack_network, input_ids, self.target_weight, content, self.annealing_scale, self.upweight_nearby).transpose(1,2) # bs seq nv
    else:#
      bs, nv, s, d = content.shape
      annealing_scores = torch.ones(bs, s, nv).to(content.device)
      #annealing_scores = 0.5 + max(0.5*(1-s/20),0)
      #annealing_scores = 1 - (math.pow(s,2)/10000)
      #annealing_scores = 0.5 + 0.5*math.cos(s*math.pi/10000)
      #annealing_scores = (0.75+0.5*torch.cos(torch.arange(s)*math.pi/(s+5)).reshape(1,1,s).to(content.device).transpose(1,2).expand(bs,s,nv))
      #annealing_scores[:2] = 0
      #import pdb; pdb.set_trace()
      #annealing_scores = torch.maximum(
      #    torch.cummin((1-torch.arange(s)/100).reshape(1,1,s).to(content.device).transpose(1,2),dim=1).values.expand(bs, s, nv),
      #    torch.tensor(0).to(content.device))

    content_weights = create_content_soft_mask(self.content_weights, input_ids, annealing_scores)
    content = content * content_weights.transpose(1,2).unsqueeze(3)

    # Compute resulting outputs
    hidden_states = torch.sum(contextualization @ content, dim=1) # (bs, s, d)
    
    lm_logits = self.backpack_network.lm_head(hidden_states)
    CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
    return CausalLMOutput(logits=lm_logits)


class NegativeWeightedBackpackLMHeadModel(nn.Module, GenerationMixin):

  def __init__(self, backpack_network, content_weights, target_weight, annealing_scale, anneal=True, upweight_nearby=True):
    super().__init__()
    self.backpack_network = backpack_network
    self.content_weights = content_weights # (nv, vocsize)
    self.target_weight = target_weight.to('cuda')
    self.annealing_scale = annealing_scale
    self.anneal = anneal
    self.upweight_nearby = upweight_nearby


  def forward(self, input_ids, position_ids=None, inference_params=None):
    """
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
    """

    # unchanged
    contextl_hidden_states = self.backpack_network.transformer.gpt2_model(input_ids, position_ids=position_ids, inference_params=inference_params)
    contextualization = self.backpack_network.transformer.contextualization_attn(contextl_hidden_states) # (bs, nv, s, s)

    # Compute content and weight
    content = self.backpack_network.transformer.content_model(input_ids, position_ids, inference_params) # (bs, nv, s, d)
    if self.anneal:
      annealing_scores = mask_annealing(self.backpack_network, input_ids, self.target_weight, content, self.annealing_scale, self.upweight_nearby).transpose(1,2) # bs seq nv
    else:
      bs, nv, s, d = content.shape
      annealing_scores = torch.ones(bs, s, nv).to(content.device)

    content_weights = create_content_soft_mask(self.content_weights, input_ids, annealing_scores)
    weighted_content = content * content_weights.transpose(1,2).unsqueeze(3)
    #import pdb; pdb.set_trace()

    #content = torch.minimum(content, weighted_content)

    # Compute resulting outputs
    #hidden_states = torch.sum(contextualization @ content, dim=1) # (bs, s, d)
    
    #lm_logits = self.backpack_network.lm_head(hidden_states)

    #content_weights = create_content_soft_mask(self.content_weights, input_ids, annealing_scores).transpose(1,2).unsqueeze(3)
    #weighted_content = content * content_weights
    #import pdb; pdb.set_trace()
    # Upweight just the most negative terms
    content_logits = content @ self.backpack_network.lm_head.weight.t()
    weighted_content_logits = weighted_content @ self.backpack_network.lm_head.weight.t()
    vocab_quantiles = torch.quantile(weighted_content_logits.float(), q=0.02, keepdim=True, dim=-1)
    content_logits = torch.where(weighted_content_logits<vocab_quantiles, weighted_content_logits, content_logits)

    #content_logits = torch.minimum(content_logits, weighted_content_logits)

    # Compute resulting outputs
    lm_logits = torch.sum(contextualization @ content_logits, dim=1) # (bs, s, d)
    
    #lm_logits = self.backpack_network.lm_head(hidden_states)
    CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
    return CausalLMOutput(logits=lm_logits)


class ReplacedWordLMHeadModel(nn.Module, GenerationMixin):

  def __init__(self, backpack_network, sense_dict):
    super().__init__()
    self.backpack_network = backpack_network
    self.sense_dict = sense_dict

  def replace_content(self, input_ids, content):
    for batch_index in range(content.shape[0]):
      for seq_index in range(content.shape[2]):
        word = input_ids[batch_index][seq_index].detach().cpu().item()
        if word in self.sense_dict:
          content[batch_index, :, seq_index, :] = self.sense_dict[word]
    return content

  def forward(self, input_ids, position_ids=None, inference_params=None):
    """
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
    """

    # unchanged
    contextl_hidden_states = self.backpack_network.transformer.gpt2_model(input_ids, position_ids=position_ids, inference_params=inference_params)
    contextualization = self.backpack_network.transformer.contextualization_attn(contextl_hidden_states) # (bs, nv, s, s)


    # Compute content and weight
    content = self.backpack_network.transformer.content_model(input_ids, position_ids, inference_params) # (bs, nv, s, d)
    content = self.replace_content(input_ids, content)

    # Compute resulting outputs
    hidden_states = torch.sum(contextualization @ content, dim=1) # (bs, s, d)
    
    lm_logits = self.backpack_network.lm_head(hidden_states)
    CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
    return CausalLMOutput(logits=lm_logits)
