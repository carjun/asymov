from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module, Transformer as T
from tqdm import tqdm
import pdb

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

def remove_padding_asymov(tensors, lengths):
    return [tensor[:, :tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

# def generate_square_subsequent_mask(sz: int) -> Tensor:
#     mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
#     return mask #[sz,sz]

def create_mask(src: Tensor, tgt: Tensor, PAD_IDX: int) -> Tuple[Tensor]:
    # src: [Frames, Batch size], tgt: [Frames-1, Batch size]
    src_seq_len = src.shape[0] #Frames
    tgt_seq_len = tgt.shape[0] #Frames-1

    tgt_mask = T.generate_square_subsequent_mask(tgt_seq_len) #[tgt_seq_len, tgt_seq_len]
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool) #[src_seq_len, src_seq_len]

    src_padding_mask = (src == PAD_IDX).transpose(0, 1) #[Batch size, Frames]
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1) #[Batch size, Frames-1]
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# function to generate output sequence using greedy algorithm
def greedy_decode(model: Module, src: Tensor, max_len: int, start_symbol: int, end_symbol: int) -> Tensor:
    # src: [Frames, 1]
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool) # [Frames, Frames]
    memory = model.encode(src, src_mask) #[Frames, 1, *]
    
    # pdb.set_trace()
    tgt = torch.ones(1, 1).fill_(start_symbol).type(torch.long)
    for i in tqdm(range(max_len-1), leave=False):
        tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))
                    .type(torch.bool))
        out = model.decode(tgt, memory, tgt_mask) #[Frames, 1, *]
        logits = model.generator(out[-1]) #[1, Classes]
        _, next_word = torch.max(logits, dim=-1)
        next_word = next_word.item()

        tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return tgt #[Frames, 1]
