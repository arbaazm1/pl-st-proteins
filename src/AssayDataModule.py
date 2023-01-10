from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader
from utils import train_val_test_split, CSVBatchedDataset

class AssayDataModule(LightningDataModule):
    def __init__(self, 
                dataset_name,
                n_train,
                batch_converter, 
                batch_size, 
                seed,
                val_split=0.2,
                test_split=0.2):
        
        super().__init__()
        self.dataset_name = dataset_name
        self.n_train = n_train
        self.batch_converter = batch_converter
        self.batch_size = batch_size
        self.seed = seed
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage):
        train_df, val_df, test_df = train_val_test_split(self.dataset_name, self.n_train, self.val_split, self.test_split, self.seed)

        if stage == "fit":
            self.train_dataset = CSVBatchedDataset.from_dataframe(train_df)
            self.train_batches = self.train_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)

            self.val_dataset = CSVBatchedDataset.from_dataframe(val_df)
            self.val_batches = self.val_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
            
        if stage == "test":
            self.test_dataset = CSVBatchedDataset.from_dataframe(test_df)
            self.test_batches = self.test_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        
        if stage == "predict":
            self.train_dataset = CSVBatchedDataset.from_dataframe(train_df)
            self.train_batches = self.train_dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)

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
        return DataLoader(self.train_dataset,
            collate_fn=self.batch_converter, 
            batch_sampler=BatchSampler(self.train_batches, self.batch_size, drop_last=False))