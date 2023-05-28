# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
from dataclasses import dataclass, field
import torch

from einops import rearrange

from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    max_sequence_len: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)


def sample(input_ids, model, max_length):
    """Sampling. This is a very simple implementation.
    We assume that all sequences in the same batch have the same length.
    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    scores = []
    with torch.inference_mode():
        logits = model(input_ids).logits[:, -1]
        scores.append(logits)
        next_token = torch.distributions.Categorical(logits=torch.log_softmax(logits,dim=-1)).sample()
        sequences = [next_token]
        seqlen = seqlen_og+1
        while seqlen < max_length:
            input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
            logits = model(input_ids).logits[:, -1]
            next_token = torch.distributions.Categorical(logits=torch.log_softmax(logits,dim=-1)).sample()
            seqlen += 1
    return SampleDecoderOnlyOutput(
        sequences=input_ids,
        scores=tuple(scores)
    )

def greedy_decode(input_ids, model, max_length):
    """Sampling. This is a very simple implementation.
    We assume that all sequences in the same batch have the same length.
    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    scores = []
    with torch.inference_mode():
        logits = model(input_ids).logits[:, -1]
        scores.append(logits)
        next_token = torch.argmax(logits, dim=-1)
        sequences = [next_token]
        seqlen = seqlen_og+1
        while seqlen < max_length:
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            logits = model(input_ids).logits[:, -1]
            next_token = torch.argmax(logits, dim=-1)
            seqlen += 1
    return SampleDecoderOnlyOutput(
        sequences=input_ids,
        scores=tuple(scores)
    )



class GenerationMixin:

    def generate(self, input_ids, max_length, return_dict_in_generate=False, output_scores=False):
        output = greedy_decode(input_ids, self, max_length)
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences

    def sample(self, input_ids, max_length, return_dict_in_generate=False, output_scores=False):
        output = sample(input_ids, self, max_length)
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences
