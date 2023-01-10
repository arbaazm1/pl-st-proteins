from esm import BatchConverter, pretrained
import pytorch_lightning as pl
import torch
from torchmetrics import SpearmanCorrCoef

from FineTunedESM import FineTunedESM
from utils import *

class SelfTrainedESM(pl.LightningModule):
    def __init__(self,
                dataset_name,
                model_path,
                n_train,
                num_self_train_iters,
                seed,
                val_split=0.2,
                test_split=0.2,
                output_dir = "/content/experiment_artifacts/",
                finetune_batch_size=512,
                finetune_learning_rate=3e-5,
                finetune_epochs=20):

        self.seed = seed 
        pl.seed_everything(self.seed)
        super(SelfTrainedESM, self).__init__()
        self.fine_tuning_model = FineTunedESM(model_path, dataset_name, finetune_batch_size, finetune_learning_rate)
        self.finetune_epochs = finetune_epochs
        
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.spearman_coef = SpearmanCorrCoef()
      
        self.save_hyperparameters()


    def forward(self, toks):
        return self.fine_tuning_model(toks)
    

    def training_step(self, batch, batch_idx):
        # y, _, x = batch
        # x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float)
        # if torch.cuda.is_available():
        #   x, y = x.cuda(), y.cuda()
        # preds = self(x)
        # mse_loss = self.mse_criterion(preds, y)
        # spearman_coef = self.spearman_coef(preds, y)
        # self.log("train_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train_spearman", spearman_coef, on_step=False, on_epoch=True, prog_bar=True)
        # return mse_loss

        
    

    def validation_step(self, batch, batch_idx):
        y, _, x = batch
        x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
          x, y = x.cuda(), y.cuda()
        preds = self(x)
        mse_loss = self.mse_criterion(preds, y)
        spearman_coef = self.spearman_coef(preds, y)
        self.log("val_mse", mse_loss)
        self.log("val_spearman", spearman_coef)
    
    def test_step(self, batch, batch_idx):
        y, _, x = batch
        x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
          x, y = x.cuda(), y.cuda()
        preds = self(x)
        mse_loss = self.mse_criterion(preds, y)
        spearman_coef = self.spearman_coef(preds, y)
        self.log("test_mse", mse_loss)
        self.log("test_spearman", spearman_coef)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)