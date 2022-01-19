from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from torchmetrics import Accuracy
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


class BertModel(pl.LightningModule):
    def __init__(self, cfg: EasyDict):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained(
            cfg.model.model_type,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False
        )

        self.tokenizer = BertTokenizer.from_pretrained(cfg.model.model_type, do_lower_case=True)
        self.cfg = cfg
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        input_ids, input_masks, token_type_ids, labels = batch

        outputs = self.model(input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=input_masks,
                             labels=labels)

        self.log("train/loss_step", outputs.loss.item())
        self.train_acc(outputs.logits.softmax(dim=-1), labels)

        return {"loss": outputs.loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.hstack([loss['loss'] for loss in outputs]).mean()
        self.log("train/loss_epoch", avg_loss.item())
        self.log("train/acc_epoch", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        input_ids, input_masks, token_type_ids, labels = batch

        outputs = self.model(input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=input_masks,
                             labels=labels)

        self.val_acc(outputs.logits.softmax(dim=-1), labels)

        return {"loss": outputs.loss}

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.hstack([loss['loss'] for loss in outputs]).mean()
        self.log("val/loss_epoch", avg_loss.item())
        val_acc = self.val_acc.compute()
        self.log("val/acc_epoch", val_acc.item())

        # That's for ModelCheckpointer, do not change
        self.log("val_acc_epoch", val_acc.item(), logger=False)

    def test_step(self, batch, batch_idx):
        input_ids, input_masks, token_type_ids = batch

        outputs = self.model(input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=input_masks)

        labels = torch.argmax(outputs.logits, dim=-1)

        return {"labels": labels}

    def test_epoch_end(self, outputs) -> None:
        test_predictions = torch.hstack([x['labels'] for x in outputs]).cpu().tolist()

        data_dir = Path(self.cfg.data.data_dir)
        current_dir = Path(self.cfg.data.current_dir)
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
        sample_submission.prediction = test_predictions

        sample_submission.to_csv(current_dir / f"submission.csv", index=False)
        print("Submission created!")

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.cfg.model.lr, eps=1e-8)
