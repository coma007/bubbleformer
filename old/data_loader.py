import zipfile
import pandas as pd
from torch.utils.data import Dataset


def extract_file(dataset_size, dataset_type):
    zip_ref = zipfile.ZipFile(f"data/MIND{dataset_size}_{dataset_type}.zip", 'r')
    zip_ref.extractall(f"data/MIND{dataset_size}_{dataset_type}")
    zip_ref.close()
    
    
def read_dataset(dataset_size, dataset_type, file_name):
    return pd.read_csv(f"data/MIND{dataset_size}_{dataset_type}/{file_name}", sep='\t', header=None)


def load_data(dataset_size, dataset_type, file_name):
    extract_file(dataset_size, dataset_type)

    data = read_dataset(dataset_size, dataset_type, file_name)
    return data


class Create_Dataset(Dataset):
    def  __init__(self, dataset_size, file_name, dataset_type, num_of_titles=None):
        dataset = load_data(dataset_size, dataset_type, file_name)
        if num_of_titles is None:
            num_of_titles = len(dataset)
        self.data = dataset[3][:num_of_titles]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataset(dataset_size, file_name, dataset_type, num_of_titles=None):
    return Create_Dataset(dataset_size, file_name, dataset_type, num_of_titles)

