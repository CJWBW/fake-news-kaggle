import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class FakeNewsKaggleDataset(Dataset):

    def __init__(self, titles, texts, labels, tokenizer, max_len):
        self.titles = titles
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, item):
        titles = str(self.titles[item])
        texts = str(self.texts[item])
        labels = self.labels[item]

        title_encoding = self.tokenizer.encode_plus(
            titles,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        text_encoding = self.tokenizer.encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'title_input_ids': title_encoding['input_ids'].flatten(),
            'attention_mask1': title_encoding['attention_mask'].flatten(),
            'text_input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask2': text_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class DataProcessor:
    TRAIN_PATH = Path(__file__).parent / "../data/train.csv"
    TEST_PATH = Path(__file__).parent / "../data/test.csv"

    state_dict = {}

    # metadata version

    @staticmethod
    def create_dataloader(titles, texts, labels, tokenizer, max_len, batch_size):
        dataset = FakeNewsKaggleDataset(
            titles=titles,
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_len=max_len
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4
        )

    @staticmethod
    def load_dataset():
        train_df_all = pd.read_csv(DataProcessor.TRAIN_PATH)
        test_df_all = pd.read_csv(DataProcessor.TEST_PATH)

        # drop when text is empty, replace with '' when title is empty, don't care about author
        train_df_all.dropna(subset=['text'])
        train_df_all.fillna('', inplace=True)
        test_df_all.dropna(subset=['text'])
        test_df_all.fillna('', inplace=True)

        # only for easier set up for running on part of the dataset
        train_df = train_df_all[:]
        test_df = test_df_all[:]

        train_titles = train_df['title']
        test_titles = test_df['title']
        train_texts = train_df['text']
        test_texts = test_df['text']
        train_labels = train_df['labels']
        test_labels = test_df['labels']

        return train_titles, test_titles, train_texts, test_texts, train_labels, test_labels

