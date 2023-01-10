import numpy as np
import os
import pandas as pd
import pathlib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchmetrics import SpearmanCorrCoef
from tqdm import tqdm

from AssayDataModule import AssayDataModule
from FineTunedESM import FineTunedESM
from utils import create_msa_df, df_to_dataloader, train_val_test_split

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

def run_experiment(
    dataset_name,
    model_path,
    n_train,
    num_self_train_iters,
    seed,
    val_split=0.2,
    test_split=0.2,
    output_dir = "/content/experiment_artifacts/",
    finetune_epochs=20,
):
    pl.seed_everything(seed)
    baseline_folder = os.path.join(output_dir, "baseline_data")
    pathlib.Path(baseline_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    mse_criterion = torch.nn.MSELoss(reduction='mean')
    spearman_coef = SpearmanCorrCoef()
    res_dict = {}
    
    baseline_model = FineTunedESM(model_path, dataset_name, 1024, 3e-5)
    checkpoint_callback = ModelCheckpoint(monitor='val_spearman', mode='max', dirpath=baseline_folder, filename="baseline_model")
    baseline_trainer = pl.Trainer(devices=1,
                        accelerator="gpu",
                        auto_lr_find=True,
                        auto_scale_batch_size="binsearch",
                        max_epochs=finetune_epochs,
                        callbacks=[checkpoint_callback])
    

    train_df, val_df, test_df = train_val_test_split(dataset_name, n_train, val_split, test_split, seed)
    train_dataloader = df_to_dataloader(train_df, baseline_model.batch_size, baseline_model.batch_converter)
    val_dataloader = df_to_dataloader(val_df, baseline_model.batch_size, baseline_model.batch_converter)

    train_actual = torch.as_tensor(train_df["log_fitness"].values)
    
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"))
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"))
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"))
    # Artifact folder should now contain train_df.csv, val_df.csv, test_df.csv

    baseline_dataloader = AssayDataModule(dataset_name, n_train, baseline_model.batch_converter, 512, seed,
                                          val_split, test_split)
    
    baseline_trainer.tune(baseline_model, datamodule=baseline_dataloader)

    baseline_trainer.fit(baseline_model, datamodule=baseline_dataloader)
    baseline_val_res = baseline_trainer.validate(baseline_model, datamodule=baseline_dataloader)[0]
    baseline_test_res = baseline_trainer.test(baseline_model, datamodule=baseline_dataloader)[0]

    baseline_train_preds = get_preds(baseline_model, train_dataloader).detach()

    res_dict["baseline_train_mse"] = mse_criterion(baseline_train_preds, train_actual)
    res_dict["baseline_train_spearman"] = spearman_coef(baseline_train_preds, train_actual)

    for k, v in baseline_val_res.items():
        res_dict[f"baseline_{k}"] = v
    
    for k, v in baseline_test_res.items():
        res_dict[f"baseline_{k}"] = v
    
    # Create MSA df with dummy pseudolabels
    msa_df = create_msa_df(dataset_name, train_df, val_df)
    msa_df[["seq", "log_fitness"]].to_csv(os.path.join(output_dir, 'msa_data.csv'), index=False)
    
    ###SELF TRAINING SETUP###
    train_mse = []
    val_mse = []
    train_spearman = []
    val_spearman = []
    best_val_spearman = None

    teacher_model = baseline_model

    ###SELF TRAINING LOOP#
    for st_iter in tqdm(range(num_self_train_iters)):
        #Get teacher_model pseudolabels for MSA sequences
        msa_df = pd.read_csv(os.path.join(output_dir, 'msa_data.csv'))
        # teacher_trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=finetune_epochs)
        msa_dataloader = df_to_dataloader(msa_df, baseline_model.batch_size, baseline_model.batch_converter)
        pseudolabels = get_preds(teacher_model, msa_dataloader).detach()
        msa_df['log_fitness'] = pseudolabels.cpu().numpy()
        #Concat pseudolabels with actual labels
        combined_labelled_df = pd.concat([msa_df[["seq", "log_fitness"]], train_df[["seq", "log_fitness"]]])
        pseudolabelled_train_dataloader = df_to_dataloader(combined_labelled_df, baseline_model.batch_size, baseline_model.batch_converter)
        #Place concatenated result in scratch folder
        # combined_labelled_df.to_csv(os.path.join(output_dir, 'combined_data.csv'))
        st_callback = ModelCheckpoint(monitor='val_spearman', mode='max', dirpath=output_dir, filename=f"st_iter_{st_iter}_model")
        st_trainer = pl.Trainer(devices=1,
                        accelerator="gpu",
                        max_epochs=finetune_epochs,
                        callbacks=[st_callback])
        student_model = FineTunedESM(model_path, dataset_name, baseline_model.batch_size, baseline_model.learning_rate)
        st_trainer.fit(student_model, 
                        train_dataloaders=pseudolabelled_train_dataloader,
                        val_dataloaders=val_dataloader)
       
        #Log train, val Spearmen + MSE for student
        train_preds = get_preds(student_model, train_dataloader).detach()
  
        if torch.cuda.is_available():
          train_actual = train_actual.cuda()
        train_mse.append(mse_criterion(train_preds, train_actual))
        train_spearman.append(spearman_coef(train_preds, train_actual))
        val_metrics = st_trainer.validate(student_model, datamodule=baseline_dataloader)[0]
        val_mse.append(val_metrics["val_mse"])
        val_spearman.append(val_metrics["val_spearman"])
        
        #Early stopping variables update
        if best_val_spearman is None or val_spearman[-1] > best_val_spearman:
            best_val_spearman = val_spearman[-1]
            best_st_model = student_model.state_dict()
            torch.save(best_st_model, os.path.join(output_dir, 'early_stopped_st_model_data.pt'))
        
        teacher_model = student_model
    ###FIN SELF TRAINING LOOP###

    #Log test Spearman from BEST MODEL STORED IN SCRATCH

    best_model = FineTunedESM(model_path,
             dataset_name,
             512,
             3e-5)
    state = torch.load(os.path.join(output_dir, 'early_stopped_st_model_data.pt'))
    best_model.load_state_dict(state)

    best_st_model_trainer = pl.Trainer(devices=1, accelerator="gpu")
    best_st_model_val_res = best_st_model_trainer.validate(best_model, datamodule=baseline_dataloader)[0]
    best_st_model_test_res = best_st_model_trainer.test(best_model, datamodule=baseline_dataloader)[0]

    best_st_model_train_preds = get_preds(best_model, train_dataloader).detach()

    res_dict["st_train_mse"] = mse_criterion(best_st_model_train_preds, train_actual)
    res_dict["st_train_spearman"] = spearman_coef(best_st_model_train_preds, train_actual)

    for k, v in best_st_model_val_res.items():
        res_dict[f"st_{k}"] = v
    
    for k, v in best_st_model_test_res.items():
        res_dict[f"st_{k}"] = v
    
    return res_dict