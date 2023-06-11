import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting import TemporalFusionTransformer, Baseline, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.data.encoders import NaNLabelEncoder, GroupNormalizer
import pytorch_forecasting

# imports for training
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# import dataset, network to train and metric to optimize
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.tuner import Tuner
import traceback

def train_model(model, train_loader, val_loader, max_epochs=100):
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1,
        weights_summary="top",
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )
    
    # fit network
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )

    return model


def load_data(path):
    # load your data
    data = pd.read_csv(path)
    
    # assuming you need to preprocess the data to be compatible with TFT
    # TODO: Insert your preprocessing code here
    # Preprocessing might involve normalizing the data, encoding categorical variables, 
    # handling missing data, and so on. It will depend on the specifics of your data.

    # split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.1)
    
    return train_data, val_data


if __name__ == "__main__":
    
    # load data
    train_data, val_data = load_data('/Users/zhezhengren/Desktop/NeuroPrior_AI/Model_Competion/EEG/Data_Processing/Data_Epoch_Range/Data_Epoch.csv')
    
    # Set the column you want to predict
    target = 'Event Name'  # replace with the actual name of the column you want to predict
    
    # Create group ids based on 'FileName'
    group_ids = ["FileName"]
    
    # TODO: Create your TimeSeriesDataSet for train and val here
    train_dataset = TimeSeriesDataSet(
        train_data,
        time_idx='Time_Point',  # replace with the column containing time information
        target=target,
        group_ids=group_ids,
        # Add additional parameters if necessary
    )
    
    val_dataset = TimeSeriesDataSet(
        val_data,
        time_idx='Time_Point',  # replace with the column containing time information
        target=target,
        group_ids=group_ids,
        # Add additional parameters if necessary
    )
  
    # create dataloaders for train and val
    batch_size = 128
    train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # create the model
    # TODO: adjust parameters to fit with your data
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset, 
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,  
        reduce_on_plateau_patience=4,
    )
    
    # train the model
    model = train_model(tft, train_loader, val_loader, max_epochs=100)
    
    # TODO: Save your model here
    torch.save(model, "tft_model.pt")
