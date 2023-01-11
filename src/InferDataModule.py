from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader
from utils import CSVBatchedDataset

class InferDataModule(LightningDataModule):
    def __init__(self, 
                df,
                batch_converter,
                batch_size):
        
        super().__init__()
        self.df = df
        self.batch_converter = batch_converter
        self.batch_size = batch_size

    def setup(self, stage):
        
        self.dataset = CSVBatchedDataset.from_dataframe(self.df)
        self.batches = self.dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)

    def train_dataloader(self):
        return DataLoader(self.dataset,
            collate_fn=self.batch_converter, batch_sampler=BatchSampler(self.batches, self.batch_size, drop_last=False))

    def predict_dataloader(self):
        return DataLoader(self.dataset,
            collate_fn=self.batch_converter, batch_sampler=BatchSampler(self.batches, self.batch_size, drop_last=False))