from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import importlib
import datetime
from config import Config
from model.NRMS import NRMS

config = Config()
Model = NRMS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    writer = SummaryWriter(
        log_dir=
        f"./runs/NRMS/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('datasets/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = Model(config, pretrained_word_embedding).to(device)

    print(model)

    dataset = BaseDataset('datasets/train/behaviors_parsed.tsv',
                          'datasets/train/news_parsed.tsv')

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True))
    criterion = nn.NLLLoss()
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ]
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    checkpoint_dir = os.path.join('./checkpoint/NRMS')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        early_stopping(checkpoint['early_stop_value'])
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        for optimizer in optimizers:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for i in tqdm(range(
            1,
            config.num_epochs * len(dataset) // config.batch_size + 1),
                  desc="Training"):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True))
            minibatch = next(dataloader)

        step += 1
        y_pred = model(minibatch["candidate_news"], minibatch["clicked_news"])

        y = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y)

        loss_full.append(loss.item())
        for optimizer in optimizers:
        	optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), step)

        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )

        if i % config.num_batches_validate == 0:
            model.eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model, 'datasets/val',
                config.num_workers, 200000)
            model.train()
            writer.add_scalar('Validation/AUC', val_auc, step)
            writer.add_scalar('Validation/MRR', val_mrr, step)
            writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
            writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
            )

            early_stop, get_better = early_stopping(-val_auc)
            if early_stop:
                tqdm.write('Early stop.')
                break
            elif get_better:
                try:
                    torch.save(
                        {
                            'model_state_dict': (model).state_dict(),
                            'optimizer_state_dict':
                            (optimizers[0]).state_dict(),
                            'step':
                            step,
                            'early_stop_value':
                            -val_auc
                        }, f"./checkpoint/NRMS/ckpt-{step}.pth")
                except OSError as error:
                    print(f"OS error: {error}")


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training NRMS')
    train()
