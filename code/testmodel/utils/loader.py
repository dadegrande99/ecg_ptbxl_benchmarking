import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import wfdb
import ast
from tqdm import tqdm

def get_dataloader(data, labels, batch_size, in_channels=12, shuffle=True):
    if isinstance(data, list):
        data = np.array(data)
    data = np.asarray(data, dtype=np.float32)  # Assicurarsi che i dati siano di tipo float32

    # Aggiungi una dimensione per i canali se necessario
    if data.ndim == 3 and data.shape[1] != in_channels:
        data = data.transpose(0, 2, 1)  # [batch_size, sequence_length, in_channels] -> [batch_size, in_channels, sequence_length]

    # Verifica che il numero di canali sia corretto
    if data.shape[1] != in_channels:
        raise ValueError(f"Expected {in_channels} channels, but got {data.shape[1]} channels")

    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def import_ptbxl(path, sampling_rate=100, clean=True):
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
        data = np.array([signal for signal, meta in data])
        return data
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    ptbxl = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    ptbxl.scp_codes = ptbxl.scp_codes.apply(lambda x: ast.literal_eval(x))
    raw = load_raw_data(ptbxl, sampling_rate, path)

    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    for el in agg_df.diagnostic_class.unique():
        ptbxl[el] = 0

    for i, row in ptbxl.iterrows():
        for key in row.scp_codes.keys():
            if key in agg_df.index:
                ptbxl.at[i, agg_df.loc[key].diagnostic_class] = 1

    # Aggiungi i dati grezzi come array numpy al DataFrame
    ptbxl['raw_data'] = list(raw)

    return raw, ptbxl

def split_data(ptbxl, folds=[8, 9, 10]):
    train_fold, val_fold, test_fold = folds
    train_df = ptbxl[ptbxl.strat_fold <= train_fold]
    val_df = ptbxl[ptbxl.strat_fold == val_fold]
    test_df = ptbxl[ptbxl.strat_fold == test_fold]
    return train_df, val_df, test_df
