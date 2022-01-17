from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from easydict import EasyDict
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataloader import ClassificationDataLoader
from src.model import BertModel

pl.seed_everything(0)


@hydra.main(config_path="configs", config_name="kaggle-gpu.yaml")
def train(cfg: DictConfig):
    cfg = EasyDict(cfg)

    current_dir = Path(get_original_cwd())
    train_df = pd.read_csv(current_dir / "data" / "train.csv")
    test_df = pd.read_csv(current_dir / "data" / "test.csv").iloc[:50]

    model = BertModel(cfg)

    dataloader = ClassificationDataLoader(
        train_df=train_df,
        test_df=test_df,
        tokenizer=model.tokenizer,
        **cfg.data
    )

    checkpointer = ModelCheckpoint(dirpath=current_dir / "models/",
                                   monitor="val/loss_epoch",
                                   mode="min",
                                   save_weights_only=True,
                                   save_top_k=1,
                                   filename=r"bert_{epoch}-{val_loss_epoch:.02f}")

    trainer = pl.Trainer(
        callbacks=[checkpointer],
        gpus=cfg.gpus,
        max_epochs=cfg.model.num_epochs,
        limit_train_batches=2,
        limit_val_batches=2,
    )

    trainer.fit(model, dataloader.train_dataloader(), dataloader.val_dataloader())
    trainer.test(model, dataloader.test_dataloader())


if __name__ == '__main__':
    train()
