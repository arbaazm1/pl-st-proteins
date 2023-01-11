from esm import BatchConverter, pretrained
import os
import pytorch_lightning as pl
import torch
from torchmetrics import SpearmanCorrCoef

from utils import read_fasta

class FineTunedESM(pl.LightningModule):

    def __init__(self, model_path, dataset_name, learning_rate):
        super().__init__()
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(model_path)
        self.mask_idx = torch.tensor(self.alphabet.mask_idx)
        if torch.cuda.is_available():
          self.mask_idx = self.mask_idx.cuda()

        wt_fasta_file = os.path.join("..", "processed_data", dataset_name, "wt.fasta")
        wt_seq = read_fasta(wt_fasta_file)[0]
        self.batch_converter = BatchConverter(self.alphabet)
        _, _, self.wt_toks = self.batch_converter([('WT', wt_seq)])
        if torch.cuda.is_available():
          self.wt_toks = self.wt_toks.cuda()
        
        self.learning_rate = learning_rate
        
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.spearman_coef = SpearmanCorrCoef()

        self.save_hyperparameters()

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

    def forward(self, toks):
        wt_toks_rep = self.wt_toks.repeat(toks.shape[0], 1)
        mask = (toks != self.wt_toks)
        masked_toks = torch.where(mask, self.mask_idx, toks)
        out = self.esm_model(masked_toks, return_contacts=False)
        logits = out["logits"]
        logits_tr = logits.transpose(1, 2)  # [B, E, T]
        ce_loss_mut = self.ce_criterion(logits_tr, toks)   # [B, E]
        ce_loss_wt = self.ce_criterion(logits_tr, wt_toks_rep)
        ll_diff_sum = torch.sum(
            (ce_loss_wt - ce_loss_mut) * mask, dim=1, keepdim=True)  # [B, 1]
        return ll_diff_sum[:, 0]
    

    def training_step(self, batch, batch_idx):
        y, _, x = batch
        x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
          x, y = x.cuda(), y.cuda()
        preds = self(x)
        mse_loss = self.mse_criterion(preds, y)
        self.log("train_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss":mse_loss, "preds":preds, "y":y}
    

    def training_epoch_end(self, training_step_outputs):
      all_preds = torch.cat([x["preds"] for x in training_step_outputs])
      all_true = torch.cat([x["y"] for x in training_step_outputs])
      if torch.cuda.is_available():
        all_preds, all_true = all_preds.cuda(), all_true.cuda()
      spearman_coef = self.spearman_coef(all_preds, all_true)
      self.log("train_spearman", spearman_coef, prog_bar=True)
    #   ndcg_train = ndcg(all_preds, all_true)
    #   self.log("train_ndcg", ndcg_train, on_step=False, on_epoch=True, prog_bar=True)
      

    def validation_step(self, batch, batch_idx):
        y, _, x = batch
        x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
          x, y = x.cuda(), y.cuda()
        preds = self(x)
        mse_loss = self.mse_criterion(preds, y)
        self.log("val_mse", mse_loss, on_step=False, on_epoch=True)
        return {"loss":mse_loss, "preds":preds, "y":y}
    

    def validation_epoch_end(self, validation_step_outputs):
      all_preds = torch.cat([x["preds"] for x in validation_step_outputs])
      all_true = torch.cat([x["y"] for x in validation_step_outputs])
      if torch.cuda.is_available():
        all_preds, all_true = all_preds.cuda(), all_true.cuda()
      spearman_coef = self.spearman_coef(all_preds, all_true)
      self.log("val_spearman", spearman_coef, on_step=False, on_epoch=True, prog_bar=True)
    #   ndcg_val = ndcg(all_preds, all_true)
    #   self.log("val_ndcg", ndcg_val, on_step=False, on_epoch=True, prog_bar=True)
    

    def test_step(self, batch, batch_idx):
        y, _, x = batch
        x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
          x, y = x.cuda(), y.cuda()
        preds = self(x).detach()
        mse_loss = self.mse_criterion(preds, y)
        self.log("test_mse", mse_loss, on_step=False, on_epoch=True)
        return {"loss":mse_loss, "preds":preds, "y":y}
    

    def test_epoch_end(self, test_step_outputs):
      all_preds = torch.cat([x["preds"] for x in test_step_outputs])
      all_true = torch.cat([x["y"] for x in test_step_outputs])
      if torch.cuda.is_available():
        all_preds, all_true = all_preds.cuda(), all_true.cuda()
      spearman_coef = self.spearman_coef(all_preds, all_true)
      self.log("test_spearman", spearman_coef, on_step=False, on_epoch=True, prog_bar=True)
    #   ndcg_test = ndcg(all_preds, all_true)
    #   self.log("test_ndcg", ndcg_test, on_step=False, on_epoch=True, prog_bar=True)
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, _, x = batch
        x = torch.as_tensor(x)
        if torch.cuda.is_available():
          x = x.cuda()
        return self(x).detach()