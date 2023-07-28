import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchtext.vocab import Vocab
# from torchtext.data.utils import get_tokenizer


from transformers import BertTokenizer, BertModel

from collections import Counter


class Bert():
    
    def __init__(self, device, titles, max_title_len):
        self.device = device
        self.titles = titles
        self.max_title_len = max_title_len
        self.init_bert()
        
        
    def init_bert(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.to(self.device)
        
        
    def embbed_titles(self):
        batch_size = 1000
        
        num_titles = len(self.titles)
        embedding_size = self.bert_model.config.hidden_size
        
        all_titles_embds = torch.empty((num_titles, self.max_title_len, embedding_size), device=self.device)  # Initialize the tensor

        # for i in range(0, num_titles, batch_size):
        #     batch_titles = self.titles[i:i + batch_size]
        #     batch_size_actual = len(batch_titles)

        #     encoded_words = self.tokenizer(batch_titles, padding='max_length', max_length=self.max_title_len, truncation=True, return_tensors="pt").to(self.device)

        #     with torch.no_grad():
        #         model_outputs = self.bert_model(**encoded_words)

        #     title_embds = model_outputs.last_hidden_state

        #     all_titles_embds[i:i + batch_size_actual] = title_embds

        for batch in self.titles:
            encoded_words = self.tokenizer(batch, padding='max_length', max_length=self.max_title_len, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                model_outputs = self.bert_model(**encoded_words)
            title_embds = model_outputs.last_hidden_state
            return title_embds
            #all_titles_embds = torch.cat((all_titles_embds, title_embds), dim=0 )

        return all_titles_embds
