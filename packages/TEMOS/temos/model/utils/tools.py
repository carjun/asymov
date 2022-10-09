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

def batch_greedy_decode(model: Module, src: Tensor, max_len: int, start_symbol: int, end_symbol: int,
                  src_mask: Tensor = None, src_padding_mask: Tensor = None) -> Tensor:
    # src: [Frames, Batches]
    if src_mask is None:
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool) # [Frames, Frames]
    
    memory = model.encode(src, src_mask, src_padding_mask) #[Frames, Batches, *]
    
    # pdb.set_trace()
    batch_size = src.shape[1]
    tgt = torch.ones(1, batch_size).fill_(start_symbol).type(torch.long) #[1, Batch size], 1 as for 1st frame
    tgt_len = torch.full((batch_size,), max_len) #[Batch Size]

    for i in tqdm(range((max_len-1)), "autoregressive translation", None):
        tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))
                    .type(torch.bool))
        if i==0:
            tgt_padding_mask = torch.full((batch_size, 1), False) #[Batch Size, 1], 1 as for 1st frame
        else:
            tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(1)], dim=1)

        out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask) #[Frames, Batch Size, *]
        logits = model.generator(out[-1]) #[Batch Size, Classes]
        next_word = torch.argmax(logits, dim=-1) #[Batch Size]
        tgt = torch.cat([tgt, next_word.unsqueeze(0)]) #[Frames+1, Batch size]
        # tgt2 = torch.argmax(model.generator(out), dim=-1)
        # assert torch.equal(tgt[1:], tgt2)
        
        tgt_len = torch.where(torch.logical_and(next_word==end_symbol, tgt_len==max_len), i+2, tgt_len)
        if (tgt_len>(i+2)).sum()==0 and (i+2)<max_len: #2nd condition is hack to run tqdm till end, not break at last index
            break
    
    tgt_list = remove_padding(tgt.permute(1, 0), tgt_len)
    return tgt_list #List[Tensor[Frames]]
