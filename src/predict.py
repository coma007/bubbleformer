import random
import numpy as np

from os import path
import torch
import importlib

from tqdm import tqdm
from evaluate import NewsDataset, UserDataset, BehaviorsDataset
from torch.utils.data import Dataset, DataLoader
from model.NRMS import NRMS
from config import Config

Model = NRMS
config = Config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sample_user(model, all_news2vector, news2vector):
    

    user_dataset = UserDataset(path.join("datasets/test", 'behaviors.tsv'),
                               'datasets/train/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    for minibatch in user_dataloader:
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([all_news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                              dim=0).transpose(0, 1)
            user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join("datasets/test", 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0

    tasks = []

    for minibatch in behaviors_dataloader:
        count += 1
        print(minibatch.keys())
        if count != 50:
            continue
        print(minibatch['impressions'])
        candidate_news_vector = torch.stack([
            news2vector[news[0].split('-')[0]]
            for news in minibatch['impressions'] 
        ], dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = torch.softmax(model.get_prediction(candidate_news_vector,
                                                 user_vector), dim=0).tolist()
        recommendations = {}
        for news, prob in zip(minibatch['impressions'], click_probability):
            recommendations[news[0].split("-")[0]] = prob, news[0].split("-")[1] == "1"
        recommendations = dict(sorted(recommendations.items(), key=lambda item: item[1], reverse=True))
        print("News ID\t\tProbability\tClicked")
        for news, (prob, clicked) in recommendations.items():
            if clicked:
                print(news, "\t\t", round(prob*100, 2), "%\t\t", clicked)
            else:
                print(news, "\t\t", round(prob*100, 2), "%\t\t")
        return sum(click_probability)

def get_news(model, news_dataloader):
    
    news2vector = {}
    for minibatch in news_dataloader:
        news_ids = minibatch["id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())
    
    return news2vector


def sample_random_news(news_vector, num_samples=100):
    news_list = list(news_vector)
    
    random_items = random.sample(news_list, num_samples)
    
    return random_items


if __name__ == '__main__':

    print(f'Predicting NRMS')

    
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(path.join('checkpoint', 'NRMS'))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    
    
    news_dataset = NewsDataset(path.join("datasets/test", 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)
    
    news_vector = get_news(model, news_dataloader)

    news_sample = sample_random_news(news_dataloader, num_samples=10)

    user_vector = sample_user(model, news_vector, get_news(model, news_sample))    
    print(user_vector)
