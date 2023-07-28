import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchtext.vocab import Vocab
# from torchtext.data.utils import get_tokenizer


from transformers import BertTokenizer, BertModel

from collections import Counter


class Bert():
    
    def __init__(self, device):
        self.device = device
        self.init_bert()
        
        
    def init_bert(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.to(self.device)
        
        
    def embbed_titles(self, titles):
        
        max_title_len = max(len(title) for title in titles)
        
        num_titles = len(titles)
        embedding_size = self.bert_model.config.hidden_size
        
        all_titles_embds = torch.empty((num_titles, max_title_len, embedding_size), device=self.device)  
        
        index = 0
        
        for batch in titles:
            encoded_words = self.tokenizer(batch, padding='max_length', max_length=max_title_len, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                model_outputs = self.bert_model(**encoded_words)
            title_embds = model_outputs.last_hidden_state
            all_titles_embds[index] = title_embds
            index += len(batch)

        return all_titles_embds
