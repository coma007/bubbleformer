import torch
import torch.nn as nn
import torch.nn.functional as F
from config import hparams
# from torchtext.vocab import Vocab
# from torchtext.data.utils import get_tokenizer


from transformers import BertTokenizer, BertModel

from collections import Counter

from bert import Bert


class NRMS(nn.Module):
    
    def __init__(self, hparams):
        super(NRMS, self).__init__()
        
        self.device = hparams['device']

        self.mha = nn.MultiheadAttention(768, 2, dropout=0.1)
        
        self.additive_attn = AdditiveAttention(hparams['max_title_len'], 768)
    
    
    def forward(self, context):
        mha_outs = []
        att_outs = []
        word_repr = []
        for title in context:
            mha_out, _ = self.mha(title, title, title)
            mha_outs.append(mha_out)
            add_out, _ = self.additive_attn(mha_out)
            att_outs.append(add_out)
            
            word_repr.append(torch.mm(torch.transpose(add_out, 0, 1), mha_out))
        
        return torch.sigmoid(word_repr)
#         return mha_outs, att_outs, word_repr


    
class AdditiveAttention(torch.nn.Module):
    def __init__(self, in_dim, v_size):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

        
    def forward(self, context):
        weights = self.proj_v(self.proj(torch.transpose(context, 0, 1))).squeeze(-1)
        weights = torch.softmax(weights, dim=-1) 
        weights = weights.unsqueeze(-1)  
        return torch.mm(context, weights), weights  
