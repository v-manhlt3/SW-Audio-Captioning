#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import torch
import torch.nn as nn
from sentence_transformers import util
import torch.nn.functional as F
import ot

import torch
from torch import nn
import math
from soft_dtw_cuda import SoftDTW
from utils_EBW import HybridEBSW, EBSW, kernel_SW

# from tools.random_prj import sliced_Wasserstein
from sentence_transformers import util

from typing import Tuple
from geomloss import SamplesLoss

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def cos_cost(a, b):
    cos_sim = util.cos_sim(a.squeeze(0),b.squeeze(0))
    dist = 1 - cos_sim
    return dist.unsqueeze(0)

@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    # dummy_x = torch.ones(x.shape[0], x.shape[1]).to(x.device)
    dummy_x = torch.clone(x)
    # dummy_x.requies_grad =True
    r_position = (dummy_x * cos) + (rotate_half(dummy_x) * sin)
    # return (x * cos) + (rotate_half(x) * sin)
    return torch.cat([x, r_position.squeeze(0).squeeze(0)], dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(
                x.shape[seq_dimension], device=x.device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype).to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def forward(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            t, seq_dimension=-2
        )

        return apply_rotary_pos_emb(t, self._cos_cached, self._sin_cached)
            # apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)].to(x.device)
        # x = torch.concat([x, self.pe[:x.size(0)].to(x.device)], dim=-1)
        # return self.dropout(x)
        return x
    
sigma_list = [1, 2, 4, 8, 16]
eps = 1e-8
def gaussian_dotprod_kernel(x, y):
    k_xx = torch.pow(x@x.t(), 2)
    k_yy = torch.pow(y@y.t(), 2)
    k_xy = 2*torch.pow(x@y.t(), 2)
    gau_kernel = torch.exp(-0.5*(k_xx + k_yy - k_xy))
    return gau_kernel




class OTBaseline(nn.Module):

    def __init__(self):
        super(OTBaseline, self).__init__()
        # self.epsilon = epsilon
        # self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        # self.use_position = use_pos
        self.pos_enc = PositionalEncoding(d_model=1024)
        # self.pos_enc = RotaryEmbedding(pos_dim)
        self.ot_dis_func = SamplesLoss(cost=cos_cost)
        # self.pos_enc = RotaryEmbedding(1024)

    def forward(self, audio_embs, audio_masks, text_embs, text_masks, useOT=True):
        batch_size = audio_embs.size(0)
        total_loss = 0

        # norm_auds = torch.nn.functional.normalize(audio_embs)
        # norm_txt = torch.nn.functional.normalize(text_embs)
        norm_auds = audio_embs
        norm_txt = text_embs
        for i in range(batch_size):
            aud_len = audio_masks[i].sum()
            text_len = text_masks[i].sum()
            aud = norm_auds[i][:aud_len.item()]
            text = norm_txt[i][:text_len.item()]
            aud = self.pos_enc(aud.unsqueeze(1)).squeeze(1)
            text = self.pos_enc(text.unsqueeze(1)).squeeze(1)
            a = torch.ones(aud.shape[0])/aud.shape[0]
            a = a.to(aud.device)
            b = torch.ones(text.shape[0])/text.shape[0]
            b = b.to(text.device)

            M_dist = util.cos_sim(aud, text)
            M_dist = 1 - M_dist

            # M_dist = ot.dist(aud, text)
            # M_dist = M_dist /M_dist.max()

            # loss = EBSW(aud, text, a, b, L=50, temp=2.0)
            # loss = self.ot_dis_func(aud, text).sum()
            # loss = loss/(aud.shape[0]*text.shape[0])
            # print("loss: ", loss)
            # loss = ot.sliced.sliced_wasserstein_distance(aud, text, n_projections=50)
            # loss = ot.sinkhorn2(a, b, M_dist, reg=0.05, numItermax=100)
            loss = ot.emd2(a, b, M_dist)
            total_loss+=loss

        return total_loss/batch_size


class OTLossAlign(nn.Module):

    def __init__(self, epsilon=0.1, use_pos=False, pos_dim=1024):
        super(OTLossAlign, self).__init__()
        self.epsilon = epsilon
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        self.use_position = use_pos
        # self.pos_enc = PositionalEncoding(d_model=pos_dim)
        self.pos_enc = RotaryEmbedding(pos_dim)

    def forward(self, audio_embs, audio_masks, text_embs, text_masks, useOT=True):
        batch_size = audio_embs.size(0)
        total_loss = 0
        if useOT:
            norm_auds = audio_embs
            norm_txt = text_embs
            for i in range(batch_size):
                aud_len = audio_masks[i].sum()
                text_len = text_masks[i].sum()
                if self.use_position:
                    ## rotary embedding
                    aud = norm_auds[i][:aud_len.item()]
                    text = norm_txt[i][:text_len.item()]
                    
                    aud = self.pos_enc(aud).squeeze(0).squeeze(0)
                    text = self.pos_enc(text).squeeze(0).squeeze(0)

                    # aud = norm_auds[i][:aud_len.item()].unsqueeze(1)
                    # text = norm_txt[i][:text_len.item()].unsqueeze(1)
                    # aud = self.pos_enc(aud).squeeze(1)
                    # text = self.pos_enc(text).squeeze(1)
                else:
                    aud = norm_auds[i][:aud_len.item()]
                    text = norm_txt[i][:text_len.item()]

                a = torch.ones(aud.shape[0])/aud.shape[0]
                a = a.to(aud.device)
                b = torch.ones(text.shape[0])/text.shape[0]
                b = b.to(text.device)
                # loss = EBSW(aud, text, a, b, L=50, temp=1.0)
                # loss = HybridEBSW(aud, text, a, b, L=50, temp=1.0)

                loss = ot.sliced.sliced_wasserstein_distance(aud, text, n_projections=50)
                total_loss+=loss

            return total_loss/batch_size
        else:
            norm_auds = torch.nn.functional.normalize(audio_embs)
            norm_txt = torch.nn.functional.normalize(text_embs)
            loss = self.sdtw(norm_auds, norm_txt)
            return loss.sum()/(norm_auds.shape[0]*norm_auds.shape[1]*norm_txt.shape[1])

class OTLossKernel(nn.Module):

    def __init__(self, epsilon=0.1, use_pos=False, pos_dim=1024):
        super(OTLossKernel, self).__init__()
        self.epsilon = epsilon
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        self.use_position = use_pos
        # self.pos_enc = PositionalEncoding(d_model=pos_dim)
        self.pos_enc = RotaryEmbedding(pos_dim)

    def forward(self, audio_embs, audio_masks, text_embs, text_masks, useOT=True):
        batch_size = audio_embs.size(0)
        total_loss = 0
        if useOT:
            norm_auds = audio_embs
            norm_txt = text_embs
            for i in range(batch_size):
                aud_len = audio_masks[i].sum()
                text_len = text_masks[i].sum()
                if self.use_position:
                    ## rotary embedding
                    aud = norm_auds[i][:aud_len.item()]
                    text = norm_txt[i][:text_len.item()]
                    
                    aud = self.pos_enc(aud).squeeze(0).squeeze(0)
                    text = self.pos_enc(text).squeeze(0).squeeze(0)

                else:
                    aud = norm_auds[i][:aud_len.item()]
                    text = norm_txt[i][:text_len.item()]

                a = torch.ones(aud.shape[0])/aud.shape[0]
                a = a.to(aud.device)
                b = torch.ones(text.shape[0])/text.shape[0]
                b = b.to(text.device)
                # loss = EBSW(aud, text, a, b, L=50, temp=1.0)
                # loss = HybridEBSW(aud, text, a, b, L=50, temp=1.0)
                # print("Audio shape: ", aud.shape)
                loss = kernel_SW(X=aud, Y=text, a=a, b=b, L=50, p=2, gamma=3.0)
                # loss = ot.sliced.sliced_wasserstein_distance(aud, text, n_projections=50)
                total_loss+=loss

            return total_loss/batch_size
        else:
            norm_auds = torch.nn.functional.normalize(audio_embs)
            norm_txt = torch.nn.functional.normalize(text_embs)
            loss = self.sdtw(norm_auds, norm_txt)
            return loss.sum()/(norm_auds.shape[0]*norm_auds.shape[1]*norm_txt.shape[1])






        
if __name__ =="__main__":
    ot_loss = OTLossAlign()
    one = torch.tensor([1,1,1])
    two = torch.tensor([2,2,2])
    three = torch.tensor([3,3,3])
    four = torch.tensor([4,4,4])
    five = torch.tensor([5,5,5])

    tensor = torch.vstack([one, two, three, four, five])
    concat_ten = ot_loss.concat(tensor)
    print(tensor.shape)
    print(concat_ten.shape)
    print(concat_ten)
    # print(concat_ten[-2,:].sum())