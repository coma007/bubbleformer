import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

class NRMS(nn.Module):
    
    def __init__(self, titles, BATCH_SIZE, device):
        super(NRMS, self).__init__()
        self.batch_size = BATCH_SIZE
        self.device = torch.device(device)
        self.titles = titles
        self.bert_model = self.init_bert(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def forward(self):
        return self.embed_titles(self.titles)
    
    @staticmethod
    def init_bert(device):
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_model.to(device)
        return bert_model
        
    def embed_titles(self, titles):
        
        title_embds = torch.empty((0, self.bert_model.config.hidden_size)).to(self.device)

        for i in range(0, len(titles), self.batch_size):
            batch_titles = titles[i:i + self.batch_size]
            encoded_inputs = self.tokenizer(batch_titles, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                model_outputs = self.bert_model(**encoded_inputs)

            batch_embds = model_outputs.last_hidden_state 
            batch_embds = batch_embds.mean(dim=1)
            title_embds = torch.cat((title_embds, batch_embds), dim=0)
        
        return title_embds
