from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from easydict import EasyDict
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.dataloader import ClassificationDataLoader
from src.model import BertModel

pl.seed_everything(0)


@hydra.main(config_path="configs", config_name="kaggle-gpu")
def train(cfg: DictConfig):
    cfg = EasyDict(cfg)

    data_dir = Path(cfg.data.data_dir)
    current_dir = Path(Path(cfg.data.current_dir))
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    model = BertModel(cfg)

    dataloader = ClassificationDataLoader(
        train_df=train_df,
        test_df=test_df,
        tokenizer=model.tokenizer,
        **cfg.data
    )

    logger = WandbLogger(name="baseline_bert",
                         save_dir=str(current_dir / "logs"),
                         log_model=True,
                         experiment="Five epochs",
                         project="contradictory-my-dear-watson",
                         config=cfg)

    logger.watch(model, log_freq=1000)

    checkpointer = ModelCheckpoint(dirpath=current_dir / "models",
                                   monitor="val/loss_epoch",
                                   mode="min",
                                   save_weights_only=True,
                                   save_top_k=1,
                                   filename=r"bert_{epoch}-{val_loss_epoch:.02f}")

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


if __name__ == '__main__':
    train()
