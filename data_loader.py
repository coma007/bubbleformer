import zipfile
import pandas as pd
from torch.utils.data import Dataset


def extract_file(dataset_size, dataset_type):
    zip_ref = zipfile.ZipFile(f"data/MIND{dataset_size}_{dataset_type}.zip", 'r')
    zip_ref.extractall(f"data/MIND{dataset_size}_{dataset_type}")
    zip_ref.close()
    
    
def read_dataset(dataset_size, dataset_type, file_name):
    return pd.read_csv(f"data/MIND{dataset_size}_{dataset_type}/{file_name}", sep='\t', header=None)


def load_data(dataset_size, file_name):
    # extract_file(dataset_size, "train")
    # extract_file(dataset_size, "dev")
#     extract_file(dataset_size, "test")
    
    data_train = read_dataset(dataset_size, "train", file_name)
    #data_dev = read_dataset(dataset_size, "dev", file_name)
    # data_test_bhv = read_dataset(dataset_size, "test", file_name)
    return data_train
#, data_dev #, data_test_bhv


class Create_Dataset(Dataset):
    def  __init__(self, dataset_size, file_name,num_of_titles):
        data_train_nws = load_data(dataset_size, file_name)
        self.data = data_train_nws[3][:num_of_titles]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataset(dataset_size, file_name,num_of_titles):
    return Create_Dataset(dataset_size, file_name,num_of_titles)

