import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ClassificationDataset(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df_data = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence1 = self.df_data.loc[index, 'premise']
        sentence2 = self.df_data.loc[index, 'hypothesis']

        # Process the sentence
        # ---------------------
        encoded_dict = self.tokenizer.encode_plus(
            sentence1, sentence2,  # Sentences to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,  # Pad or truncate all sentences.
            padding="max_length",
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # These are torch tensors already.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        # Convert the target to a torch tensor
        target = torch.tensor(self.df_data.loc[index, 'label'])

        sample = (padded_token_list, att_mask, token_type_ids, target)

        return sample

    def __len__(self):
        return len(self.df_data)


class ClassificationTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df_data = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence1 = self.df_data.loc[index, 'premise'].__str__()
        sentence2 = self.df_data.loc[index, 'hypothesis'].__str__()

        # Process the sentence
        # ---------------------
        encoded_dict = self.tokenizer.encode_plus(
            sentence1, sentence2,  # Sentences to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,  # Pad or truncate all sentences.
            padding="max_length",
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # These are torch tensors already.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        sample = (padded_token_list, att_mask, token_type_ids)

        return sample

    def __len__(self):
        return len(self.df_data)


class ClassificationDataLoader(pl.LightningDataModule):
    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 tokenizer,
                 max_len: int,
                 val_size: float,
                 batch_size: int,
                 num_workers: int):
        super().__init__()
        num_val_samples = int(len(train_df) * val_size)
        num_train_samples = len(train_df) - num_val_samples

        self.train_dataset, self.val_dataset = random_split(
            ClassificationDataset(df=train_df, tokenizer=tokenizer, max_length=max_len),
            lengths=[num_train_samples, num_val_samples],
            generator=torch.Generator().manual_seed(0)
        )

        self.test_dataset = ClassificationTestDataset(df=test_df, tokenizer=tokenizer, max_length=max_len)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
