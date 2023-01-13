import json
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
from utils import create_msa_df, df_to_dataloader, get_pl_preds, train_val_test_split


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
    initial_lr=3e-7
):
    pl.seed_everything(seed)
    baseline_folder = os.path.join(output_dir, "baseline_data")
    pathlib.Path(baseline_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    mse_criterion = torch.nn.MSELoss(reduction='mean')
    spearman_coef = SpearmanCorrCoef()
    res_dict = {}
    
    baseline_model = FineTunedESM(model_path, dataset_name, initial_lr)
    checkpoint_callback = ModelCheckpoint(monitor='val_spearman', mode='max', dirpath=baseline_folder, filename="baseline_model")
    baseline_trainer = pl.Trainer(accelerator="auto",
                        auto_lr_find=True,
                        auto_scale_batch_size="binsearch",
                        max_epochs=finetune_epochs,
                        callbacks=[checkpoint_callback])
    
    baseline_datamodule = AssayDataModule(dataset_name, n_train, baseline_model.batch_converter, 512, seed,
                                          val_split, test_split)
    

    train_df, val_df, test_df = train_val_test_split(dataset_name, n_train, val_split, test_split, seed)
    train_dataloader = df_to_dataloader(train_df, baseline_datamodule.batch_size, baseline_model.batch_converter)
    val_dataloader = df_to_dataloader(val_df, baseline_datamodule.batch_size, baseline_model.batch_converter)

    train_actual = torch.as_tensor(train_df["log_fitness"].values)
    
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"))
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"))
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"))
    # Artifact folder should now contain train_df.csv, val_df.csv, test_df.csv
    
    baseline_trainer.tune(baseline_model, datamodule=baseline_datamodule)

    baseline_trainer.fit(baseline_model, datamodule=baseline_datamodule)
    baseline_val_res = baseline_trainer.validate(baseline_model, datamodule=baseline_datamodule)[0]
    baseline_test_res = baseline_trainer.test(baseline_model, datamodule=baseline_datamodule)[0]

    baseline_train_preds = torch.cat(baseline_trainer.predict(baseline_model, datamodule=baseline_datamodule))

    res_dict["baseline_train_mse"] = mse_criterion(baseline_train_preds, train_actual)
    res_dict["baseline_train_spearman"] = spearman_coef(baseline_train_preds, train_actual)

    for k, v in baseline_val_res.items():
        res_dict[f"baseline_{k}"] = v
    
    for k, v in baseline_test_res.items():
        res_dict[f"baseline_{k}"] = v
    
    for k, v in res_dict.items():
      if isinstance(v, torch.Tensor):
        res_dict[k] = v.cpu().tolist()
    with open(os.path.join(output_dir, f"baseline_result_metrics.json"), "w") as outfile:
        json.dump(res_dict, outfile)
    
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

    eval_batch_size = None
    ###SELF TRAINING LOOP#
    for st_iter in tqdm(range(num_self_train_iters)):
        #Get teacher_model pseudolabels for MSA sequences
        msa_df = pd.read_csv(os.path.join(output_dir, 'msa_data.csv'))
        # teacher_trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=finetune_epochs)
        # msa_dataloader = df_to_dataloader(msa_df, baseline_model.batch_size, baseline_model.batch_converter)
        if eval_batch_size is None:
            pseudolabels, eval_batch_size = get_pl_preds(teacher_model, msa_df, 
                                                        baseline_model.batch_converter, eval_batch_size)
        else:
            pseudolabels, _ = get_pl_preds(teacher_model, msa_df, 
                                        baseline_model.batch_converter, eval_batch_size)
        msa_df['log_fitness'] = pseudolabels.cpu().numpy()
        #Concat pseudolabels with actual labels
        combined_labelled_df = pd.concat([msa_df[["seq", "log_fitness"]], train_df[["seq", "log_fitness"]]])
        pseudolabelled_train_dataloader = df_to_dataloader(combined_labelled_df, baseline_datamodule.batch_size, baseline_model.batch_converter)
        #Place concatenated result in scratch folder

        st_callback = ModelCheckpoint(monitor='val_spearman', mode='max')
        st_trainer = pl.Trainer(accelerator="auto",
                        max_epochs=finetune_epochs,
                        callbacks=[st_callback])
        student_model = FineTunedESM(model_path, dataset_name, baseline_model.learning_rate)
        st_trainer.fit(student_model, 
                        train_dataloaders=pseudolabelled_train_dataloader,
                        val_dataloaders=val_dataloader)
       
        #Log train, val Spearmen + MSE for student
        train_preds = torch.cat(st_trainer.predict(student_model, datamodule=baseline_datamodule))
  
        # if torch.cuda.is_available():
        #   train_actual = train_actual.cuda()
        #   train_preds = train_preds.cuda()
        
        train_mse.append(mse_criterion(train_preds, train_actual))
        train_spearman.append(spearman_coef(train_preds, train_actual))
        val_metrics = st_trainer.validate(student_model, datamodule=baseline_datamodule)[0]
        val_mse.append(val_metrics["val_mse"])
        val_spearman.append(val_metrics["val_spearman"])

        np.savetxt(os.path.join(output_dir, 'mse_trajectory_train.npy'), train_mse)
        np.savetxt(os.path.join(output_dir, 'mse_trajectory_val.npy'), val_mse)
        np.savetxt(os.path.join(output_dir, 'spearman_trajectory_val.npy'), train_spearman)
        np.savetxt(os.path.join(output_dir, 'spearman_trajectory_val.npy'), val_spearman)
        
        #Early stopping variables update
        if best_val_spearman is None or val_spearman[-1] > best_val_spearman:
            best_val_spearman = val_spearman[-1]
            best_st_model = student_model.state_dict()
            torch.save(best_st_model, os.path.join(output_dir, 'early_stopped_st_model_data.pt'))
            np.savetxt(os.path.join(output_dir, 'best_val_spearman.npy'), [best_val_spearman])
        
        teacher_model = student_model
    ###FIN SELF TRAINING LOOP###

    #Log test Spearman from BEST MODEL STORED IN SCRATCH

    best_model = FineTunedESM(model_path,
             dataset_name,
             None)
    state = torch.load(os.path.join(output_dir, 'early_stopped_st_model_data.pt'))
    best_model.load_state_dict(state)

    best_st_model_trainer = pl.Trainer(accelerator="auto")
    best_st_model_val_res = best_st_model_trainer.validate(best_model, datamodule=baseline_datamodule)[0]
    best_st_model_test_res = best_st_model_trainer.test(best_model, datamodule=baseline_datamodule)[0]

    best_st_model_train_preds = torch.cat(best_st_model_trainer.predict(best_model, datamodule=baseline_datamodule))

    res_dict["st_train_mse"] = mse_criterion(best_st_model_train_preds, train_actual).item()
    res_dict["st_train_spearman"] = spearman_coef(best_st_model_train_preds, train_actual).item()

    for k, v in best_st_model_val_res.items():
        res_dict[f"st_{k}"] = v
    
    for k, v in best_st_model_test_res.items():
        res_dict[f"st_{k}"] = v
    
    with open(os.path.join(output_dir, f"experiment_result_metrics.json"), "w") as outfile:
        json.dump(res_dict, outfile)
    
    return res_dict