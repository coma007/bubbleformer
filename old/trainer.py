import pytorch_ranger
import torch
from torch.utils import data

import pytorch_lightning as pl
from pytorch_ranger import Ranger
from model import NRMS
from metric import ndcg_score, mrr_score
import data_loader
from bert import Bert

from config import hparams

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        
        self.model = NRMS(hparams)

        self.bert = Bert(hparams['device'])

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=hparams['lr'], weight_decay=1e-5)
        return pytorch_ranger.Ranger(self.parameters(), lr=hparams['lr'], weight_decay=1e-5)

    def prepare_data(self):
        
        self.train_ds =  self.bert.embbed_titles(data_loader.get_dataset("small", "news.tsv", "train"))
        self.val_ds = self.bert.embbed_titles(data_loader.get_dataset("small", "news.tsv", "dev"))
        
        self.logger.experiment.add_graph(self.model, tmp)
        # num_train = int(len(ds) * 0.85)
        # num_val = len(ds) - num_train
        # self.train_ds, self.val_ds = data.random_split(ds, (num_train, num_val))

    def train_dataloader(self):
        
        return data.DataLoader(self.train_ds, batch_size=hparams['batch_size'], num_workers=10, shuffle=True)

    
    def val_dataloader(self):
        
        sampler = data.RandomSampler(
            self.val_ds, num_samples=10000, replacement=True)
        return data.DataLoader(self.val_ds, sampler=sampler, batch_size=hparams['batch_size'], num_workers=10, drop_last=True)
    

    def forward(self):
        """forward
        define as normal pytorch model
        """
        return None

    def training_step(self, batch, batch_idx):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        titles = batch
        loss = self.model(titles)
        return {'loss': loss}

    def on_train_epoch_end(self, outputs):
        """for each epoch end

        Arguments:
            outputs: list of training_step output
        """
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'train_loss': loss_mean}
        self.model.eval()

        # self.logger.log_metrics(logs, self.current_epoch)
        return {'progress_bar': logs, 'log': logs}

    def validation_step(self, batch, batch_idx):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        clicks, cands, cands_label = batch
        with torch.no_grad():
            logits = self.model(clicks, cands)
        mrr = 0.
        auc = 0.
        ndcg5, ndcg10 = 0., 0.

        for score, label in zip(logits, cands_label):
            auc += pl.metrics.functional.auroc(score, label)
            score = score.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            mrr += mrr_score(label, score)
            ndcg5 += ndcg_score(label, score, 5)
            ndcg10 += ndcg_score(label, score, 10)
        return {'auroc': (auc / logits.shape[0]).item(), 'mrr': (mrr / logits.shape[0]).item(), 'ndcg5': (ndcg5 / logits.shape[0]).item(), 'ndcg10': (ndcg10 / logits.shape[0]).item()}

    def on_validation_epoch_end(self, outputs):
        """
        validation end

        Arguments:
            outputs: list of training_step output
        """
        mrr = torch.tensor([x['mrr'] for x in outputs])
        auroc = torch.tensor([x['auroc'] for x in outputs])
        ndcg5 = torch.tensor([x['ndcg5'] for x in outputs])
        ndcg10 = torch.tensor([x['ndcg10'] for x in outputs])

        logs = {'auroc': auroc.mean(), 'mrr': mrr.mean(
        ), 'ndcg@5': ndcg5.mean(), 'ndcg@10': ndcg10.mean()}
        # self.logger.log_metrics(logs, self.current_epoch)
        self.model.train()
        return {'progress_bar': logs, 'log': logs}
