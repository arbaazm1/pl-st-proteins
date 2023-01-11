from Bio import SeqIO
from esm import FastaBatchedDataset
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.utils import shuffle
import torch

def read_fasta(filename, return_ids=False):
    records = SeqIO.parse(filename, 'fasta')
    seqs = list()
    ids = list()
    for record in records:
        seqs.append(str(record.seq))
        ids.append(str(record.id))
    if return_ids:
        return seqs, ids
    else:
        return seqs

class CSVBatchedDataset(FastaBatchedDataset):
    @classmethod
    def from_file(cls, csv_file):
        df = pd.read_csv(csv_file)
        return cls(df.log_fitness.values, df.seq.values)

    @classmethod
    def from_dataframe(cls, df):
        return cls(df.log_fitness.values, df.seq.values)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return super().__getitem__(idx)
        elif isinstance(idx, list):
            return super().__getitem__(idx[0])

def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation

def ndcg(y_pred, y_true):
    y_true_normalized = (y_true - y_true.mean()) / y_true.std()
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

def train_val_test_split(dataset_name, n_train, val_split, test_split, seed):
    data_path = os.path.join("..", "processed_data", dataset_name, "data.csv")
    df_full = shuffle(pd.read_csv(data_path), random_state=seed)

    n_test = int(len(df_full) * test_split)
    df_test = df_full[-n_test:]
    df_trainval = df_full.drop(df_test.index)

    if n_train == -1:
        n_train = int(len(df_trainval))
    if n_train > len(df_trainval):
        print(f"Insufficient data")
        return
    n_val = int(n_train * val_split)
    df_train = df_trainval[:n_train-n_val]
    df_val = df_trainval[n_train-n_val:n_train]

    return df_train, df_val, df_test

def create_msa_df(dataset_name, train_df, val_df):
    dataset_prefix = '_'.join(dataset_name.split('_')[:2])
    msa_a2m_path = os.path.join("..", "alignments", f"{dataset_prefix}.a2m")
    
    names, seqs = [], []

    for record in SeqIO.parse(msa_a2m_path, "fasta"):
        names.append(record.id)
        seqs.append(str(record.seq))

    alignment_df = pd.DataFrame()
    alignment_df["mutant"] = names
    alignment_df["seq"] = pd.Series(seqs).apply(lambda x: x.upper().replace(' ', ''))
    alignment_df["log_fitness"] = [0 for seq in alignment_df["seq"]]

    #Remove repeated MSA sequences
    alignment_df.drop_duplicates(subset=["seq"],inplace=True)

    #Remove MSA sequences that have labels
    labelled_train_seqs = alignment_df["seq"].isin(train_df["seq"])
    msa_train_filtered_df = alignment_df[~labelled_train_seqs]
    labelled_val_seqs = msa_train_filtered_df["seq"].isin(val_df["seq"])
    msa_filtered_df = msa_train_filtered_df[~labelled_val_seqs]

    return msa_filtered_df

def df_to_dataloader(df, batch_size, batch_converter):
    dataset = CSVBatchedDataset.from_dataframe(df)
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    
    return torch.utils.data.DataLoader(dataset,
            collate_fn=batch_converter, batch_sampler=batches)

def get_preds(model, data_loader):
    model.eval()
    y_pred = torch.Tensor()
    if torch.cuda.is_available():
      model.cuda()
      y_pred.cuda()
    with torch.no_grad():
        for batch_idx, (labels, _, toks) in enumerate(data_loader):
            if torch.cuda.is_available():
              toks = toks.cuda()
            predictions = model(toks)
            if torch.cuda.is_available():
              y_pred = y_pred.cuda()
              predictions = predictions.cuda()
            y_pred = torch.cat((y_pred, predictions))
            
    return torch.flatten(y_pred)

def get_pl_preds(model, df, batch_converter, batch_size):
    from InferDataModule import InferDataModule

    if batch_size is None:
        infer_data_module = InferDataModule(df, batch_converter, 512) 
        pl_predictor = pl.Trainer(devices=1,
                            accelerator="gpu",
                            auto_scale_batch_size="binsearch")
        pl_predictor.tune(model, datamodule=infer_data_module)
    else:
        infer_data_module = InferDataModule(df, batch_converter, batch_size) 
        pl_predictor = pl.Trainer(devices=1,
                            accelerator="gpu")

    return torch.cat(pl_predictor.predict(model, datamodule=infer_data_module)), infer_data_module.batch_size