import zipfile
import pandas as pd


def extract_file(dataset_size, dataset_type):
    zip_ref = zipfile.ZipFile(f"data/MIND{dataset_size}_{dataset_type}.zip", 'r')
    zip_ref.extractall(f"data/MIND{dataset_size}_{dataset_type}")
    zip_ref.close()
    
    
def read_dataset(dataset_size, dataset_type, file_name):
    return pd.read_csv(f"data/MIND{dataset_size}_{dataset_type}/{file_name}", sep='\t', header=None)


def load_data(dataset_size, file_name):
    extract_file(dataset_size, "train")
    extract_file(dataset_size, "dev")
#     extract_file(dataset_size, "test")
    
    data_train_bhv = read_dataset(dataset_size, "train", file_name)
    data_dev_bhv = read_dataset(dataset_size, "dev", file_name)
    # data_test_bhv = read_dataset(dataset_size, "test", file_name)
    return data_train_bhv, data_dev_bhv #, data_test_bhv