import pdb
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from pytorch_lightning import LightningModule
from temos.model.utils import PositionalEncoding

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        #TODO: padding_idx for embedding
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(LightningModule):
    def __init__(self,
                 span: bool,
                 traj: bool,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 mw_gen: list = [],
                 traj_seq: list = [],
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        
        #mw generator:
        if mw_gen and len(mw_gen)>0:
            mw_gen.insert(0, emb_size)
            mw_gen.insert(len(mw_gen), tgt_vocab_size)
            temp_gen = []
            for n,m in zip(mw_gen[:-1], mw_gen[1:]):
                temp_gen += [nn.Linear(n,m)]
            self.generator = nn.Sequential(*temp_gen)
        else:
            self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # pdb.set_trace()

        if span: 
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size-1)
            self.tgt_positional_encoding = PositionalEncoding(emb_size-1, dropout=dropout)
            
            self.span_generator = nn.Linear(emb_size, 1)

        # TODO check how to handle traj and span together

            if traj:
                self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size-3)
                self.tgt_positional_encoding = PositionalEncoding(emb_size-3, dropout=dropout)
                
                #traj generator:
                if traj_seq and len(traj_seq)>0:
                    traj_seq.insert(0, emb_size)
                    traj_seq.insert(len(traj_seq), 3)
                    temp_traj_gen = []
                    for n,m in zip(traj_seq[:-1], traj_seq[1:]):
                        temp_traj_gen += [nn.Linear(n,m)]
                    self.traj_generator = nn.Sequential(*temp_traj_gen)
                else:
                    self.traj_generator = nn.Linear(emb_size, 3)
        else:
            self.tgt_positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.src_positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                tgt_span: Tensor = None,
                tgt_traj: Tensor = None):
        # pdb.set_trace()
        src_emb = self.src_positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.tgt_positional_encoding(self.tgt_tok_emb(tgt))
        if self.hparams.traj:
            assert tgt_traj is not None
            
            # tgt_emb = self.tgt_tok_emb(tgt)
            assert tgt_traj.shape[:-1] == tgt_emb.shape[:-1] and tgt_traj.shape[-1] == 3
            
            tgt_emb = torch.cat((tgt_emb, tgt_traj), -1)
            # tgt_emb = self.positional_encoding(tgt_emb)
        # else:
        #     tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        if self.hparams.span:
            tgt_emb = torch.cat((tgt_emb, tgt_span), -1)
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        if self.hparams.span:
            return self.generator(outs), self.span_generator(outs)
        if self.hparams.traj:                                       #TODO make them compatible with each other
            return self.generator(outs), self.traj_generator(outs)
        else:
            return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor = None):
        return self.transformer.encoder(self.src_positional_encoding(self.src_tok_emb(src)), 
                                        src_mask, src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, 
               tgt_mask: Tensor, memory_mask: Tensor = None, tgt_padding_mask: Tensor = None, memory_key_padding_mask: Tensor = None,
               tgt_traj: Tensor = None, tgt_span: Tensor = None):
        tgt_emb = self.tgt_positional_encoding(self.tgt_tok_emb(tgt))
        if self.hparams.traj:
            assert tgt_traj is not None
            
            # tgt_emb = self.tgt_tok_emb(tgt)
            assert tgt_traj.shape[:-1] == tgt_emb.shape[:-1] and tgt_traj.shape[-1] == 3
            
            tgt_emb = torch.cat((tgt_emb, tgt_traj), -1)
            # tgt_emb = self.positional_encoding(tgt_emb)
        if self.hparams.span:
            tgt_emb = torch.cat((tgt_emb, tgt_span), -1)
        # else:
        #     tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        
        return self.transformer.decoder(tgt_emb, memory,
                                        tgt_mask, memory_mask, tgt_padding_mask, memory_key_padding_mask)