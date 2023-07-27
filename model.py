import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer


from transformers import BertTokenizer, BertModel

from collections import Counter

from bert import Bert


class NRMS(nn.Module):
    
    def __init__(self, device, titles):
        super(NRMS, self).__init__()
        self.device = device
        self.titles = titles
        
        bert = Bert(device, titles)
        
        self.titles_embbedings = bert.embbed_titles()
        self.mha = nn.MultiheadAttention(768, 2, dropout=0.1)
        
        self.additive_attn = AdditiveAttention(14, 768)
    
    
    def forward(self):
        mha_outs = []
        att_outs = []
        for title in self.titles_embbedings:
            title_output, weights = self.mha(title, title, title)
            mha_outs.append(title_output)
            
            title_repr, _ = self.additive_attn(title_output)
#             torch.sigmoid(title_repr)
            att_outs.append(title_repr)
        return mha_outs, att_outs


    
class AdditiveAttention(torch.nn.Module):
    def __init__(self, in_dim=100, v_size=768):
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
