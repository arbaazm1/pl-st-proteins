from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils import CSVBatchedDataset

class DFDataModule(LightningDataModule):
    def __init__(self, 
                train_df=None,
                val_df=None,
                test_df=None,
                predict_df=None,
                batch_converter=None, 
                batch_size=None):
        
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.predict_df = predict_df
        self.batch_converter = batch_converter
        self.batch_size = batch_size

    def setup(self, stage):

        if self.train_df is not None:
            self.train_dataset = CSVBatchedDataset.from_dataframe(self.train_df)
            self.train_batches = self.train_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        
        if self.val_df is not None:
            self.val_dataset = CSVBatchedDataset.from_dataframe(self.val_df)
            self.val_batches = self.val_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        
        if self.test_df is not None:
            self.test_dataset = CSVBatchedDataset.from_dataframe(self.test_df)
            self.test_batches = self.test_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        
        if self.predict_df is not None:
            self.predict_dataset = CSVBatchedDataset.from_dataframe(self.predict_df)
            self.predict_batches = self.predict_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
            collate_fn=self.batch_converter, batch_sampler=self.train_batches)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
            collate_fn=self.batch_converter, batch_sampler=self.val_batches)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
            collate_fn=self.batch_converter, batch_sampler=self.test_batches)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
            collate_fn=self.batch_converter, batch_sampler=self.predict_batches)
