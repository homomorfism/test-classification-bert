from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import wandb
from easydict import EasyDict
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from src.dataloader import ClassificationDataLoader
from src.model import BertModel

pl.seed_everything(0)


@hydra.main(config_path="configs", config_name="kaggle-gpu")
def train(cfg: DictConfig):
    cfg = EasyDict(cfg)

    data_dir = Path(cfg.data.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_data, val_data = train_test_split(train_df, test_size=cfg.data.val_size)
    train_model(cfg, train_data, val_data, test_df)

    print("training is finished!")


def train_model(cfg: DictConfig,
                train_data: pd.DataFrame,
                val_data: pd.DataFrame,
                test_data: pd.DataFrame):
    model = BertModel(cfg)

    dataloader = ClassificationDataLoader(
        train_df=train_data.reset_index(drop=True),
        val_df=val_data.reset_index(drop=True),
        test_df=test_data,
        tokenizer=model.tokenizer,
        **cfg.data
    )

    current_dir = Path(cfg.data.current_dir)
    logger = WandbLogger(name=cfg.exp_name,
                         save_dir=str(current_dir / "logs"),
                         log_model='all',
                         project="contradictory-my-dear-watson",
                         config=cfg)

    checkpointer = ModelCheckpoint(dirpath=current_dir / "models",
                                   monitor="val/loss_epoch",
                                   mode="min",
                                   save_weights_only=True,
                                   save_top_k=1,
                                   filename=r"bert_{epoch}-{val_loss_epoch:.02f}_fold=")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpointer],
        gpus=cfg.gpus,
        max_epochs=cfg.model.num_epochs,
        default_root_dir=str(current_dir / "logs"),
        # limit_train_batches=2,
        # limit_val_batches=2,
    )

    trainer.fit(model, dataloader.train_dataloader(), dataloader.val_dataloader())
    trainer.test(model, dataloader.test_dataloader())
    wandb.finish()


if __name__ == '__main__':
    train()
