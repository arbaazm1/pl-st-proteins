import os
import pandas as pd
import pathlib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchmetrics import SpearmanCorrCoef
from tqdm import tqdm

from AssayDataModule import AssayDataModule
from DFDataModule import DFDataModule
from FineTunedESM import FineTunedESM
from utils import create_msa_df, df_to_dataloader, train_val_test_split

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
    
    baseline_model = FineTunedESM(model_path, dataset_name, 1024, 3e-5)
    checkpoint_callback = ModelCheckpoint(monitor='val_spearman', mode='max', dirpath=baseline_folder, filename="baseline_model")
    baseline_trainer = pl.Trainer(devices=1,
                        accelerator="gpu",
                        auto_lr_find=True,
                        auto_scale_batch_size="binsearch",
                        max_epochs=finetune_epochs,
                        callbacks=[checkpoint_callback])
    

    baseline_dataloader = AssayDataModule(dataset_name, n_train, baseline_model.batch_converter, 512, seed)
    
    baseline_trainer.tune(baseline_model, datamodule=baseline_dataloader)

    baseline_trainer.fit(baseline_model, datamodule=baseline_dataloader)
    baseline_trainer.validate(baseline_model, datamodule=baseline_dataloader)
    baseline_trainer.test(baseline_model, datamodule=baseline_dataloader)

    train_df, val_df, test_df = train_val_test_split(dataset_name, n_train, val_split, test_split, seed)
    original_split_dataloader = DFDataModule(train_df=train_df, val_df=val_df, test_df=test_df, predict_df=train_df,
                                            batch_converter=baseline_model.batch_converter, batch_size=baseline_model.batch_size)
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"))
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"))
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"))
    # Artifact folder should now contain train_df.csv, val_df.csv, test_df.csv
    
    # Create MSA df with dummy pseudolabels
    msa_df = create_msa_df(dataset_name, train_df, val_df)
    msa_df[["seq", "log_fitness"]].to_csv(os.path.join(output_dir, 'msa_data.csv'), index=False)
    
    ###SELF TRAINING SETUP###
    train_mse = []
    val_mse = []
    train_spearman = []
    val_spearman = []
    best_val_spearman = None

    teacher_model = FineTunedESM(model_path, dataset_name, baseline_model.batch_size, baseline_model.learning_rate)

    ###SELF TRAINING LOOP#
    for st_iter in tqdm(range(num_self_train_iters)):
        #Get teacher_model pseudolabels for MSA sequences
        msa_df = pd.read_csv(os.path.join(output_dir, 'msa_data.csv'))
        teacher_trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=finetune_epochs)
        msa_dataloader = DFDataModule(predict_df=msa_df, batch_size=baseline_model.batch_size, batch_converter=baseline_model.batch_converter)
        pseudolabels = teacher_trainer.predict(teacher_model, datamodule=msa_dataloader)
        msa_df['log_fitness'] = pseudolabels
        #Concat pseudolabels with actual labels
        combined_labelled_df = pd.concat([msa_df[["seq", "log_fitness"]], train_df[["seq", "log_fitness"]]])
        pseudolabelled_train_dataloader = DFDataModule(train_df=combined_labelled_df, val_df=val_df, 
                                                        batch_size=baseline_model.batch_size, batch_converter=baseline_model.batch_converter)
        #Place concatenated result in scratch folder
        # combined_labelled_df.to_csv(os.path.join(output_dir, 'combined_data.csv'))
        st_callback = ModelCheckpoint(monitor='val_spearman', mode='max', dir_path=output_dir, filename=f"st_iter_{st_iter}_model")
        st_trainer = pl.Trainer(devices=1,
                        accelerator="gpu",
                        max_epochs=finetune_epochs,
                        callbacks=[st_callback])
        student_model = FineTunedESM(model_path, dataset_name, baseline_model.batch_size, baseline_model.learning_rate)
        st_trainer.fit(student_model, datamodule=pseudolabelled_train_dataloader)
       
        #Log train, val Spearmen + MSE for student
        train_mse.append(mse_criterion(st_trainer.predict(student_model, datamodule=original_split_dataloader), train_df["log_fitness"]))
        train_spearman.append(spearman_coef(st_trainer.predict(student_model, datamodule=original_split_dataloader), train_df["log_fitness"]))
        val_mse.append(st_trainer.validate(student_model, datamodule=baseline_dataloader)["val_mse"])
        val_spearman.append(st_trainer.validate(student_model, datamodule=baseline_dataloader)["val_spearman"])
        
        #Early stopping variables update
        if best_val_spearman is None or val_spearman[-1] > best_val_spearman:
            best_val_spearman = val_spearman[-1]
            best_st_model = student_model.state_dict()
            torch.save(best_st_model, os.path.join(output_dir, 'early_stopped_st_model_data.pt'))
        
        teacher_model = student_model
    ###FIN SELF TRAINING LOOP###

    #Log test Spearman from BEST MODEL STORED IN SCRATCH
    best_st_model_trainer = pl.Trainer(devices=1, accelerator="gpu")
    best_st_model_val_res = best_st_model_trainer.validate(ckpt_path=os.path.join(output_dir, 'early_stopped_st_model_data.pt'), datamodule=baseline_dataloader)
    best_st_model_test_res = best_st_model_trainer.test(ckpt_path=os.path.join(output_dir, 'early_stopped_st_model_data.pt'), datamodule=baseline_dataloader)
    
    res_dict = {}

    best_st_model_train_preds = best_st_model_trainer.predict(ckpt_path=os.path.join(output_dir, 'early_stopped_st_model_data.pt'), dataloaders=train_dataloader)

    res_dict["train_mse"] = mse_criterion(best_st_model_train_preds, train_df["log_fitness"])
    res_dict["train_spearman"] = spearman_coef(best_st_model_train_preds, train_df["log_fitness"])

    for k, v in best_st_model_val_res:
        res_dict[k] = v
    
    for k, v in best_st_model_test_res:
        res_dict[k] = v
    
    return res_dict