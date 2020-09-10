import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.utils.data import random_split


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

        # from the analysis below, text can go up to 24234 words, too big for analyzing
        # can use 128 words from text but mainly will use titles, which has maximum length of 72 words

        # drop when title is empty, replace with '' when text is empty, don't care about author
        train_df_all = train_df_all[train_df_all['title'].notna()]
        test_df_all = test_df_all[test_df_all['title'].notna()]

        # print(train_df_all.isnull().sum())
        train_df_all.fillna('', inplace=True)
        test_df_all.fillna('', inplace=True)

        # split train to train and validation
        total_train_size = len(train_df_all['title'].values)
        new_train_size = int(0.9 * total_train_size)
        new_train_df_all, validation_df_all = random_split(train_df_all, [new_train_size, total_train_size - new_train_size],
                                                               generator=torch.Generator().manual_seed(42))

        # actual execution part of the data, to avoid out of memory, only for easier set up for running on part of the dataset
        new_train_df = new_train_df_all.dataset[:10]
        test_df = test_df_all[:1]
        validation_df = validation_df_all.dataset[:1]

        train_titles = new_train_df['title'].values
        test_titles = test_df['title'].values
        validation_titles = validation_df['title'].values
        train_texts = new_train_df['text'].values
        test_texts = test_df['text'].values
        validation_texts = validation_df['text'].values
        train_labels = new_train_df['label'].values
        validation_labels = validation_df['label'].values
        test_ids = test_df['id'].values

        '''
        # only for analyzing data
        # id, length
        max_train_text = (-1, 0)
        max_test_text = (-1, 0)
        max_train_title = (-1, 0)
        max_test_title = (-1, 0)

        for _, train in train_df_all.iterrows():
            if len(train['text'].split()) > max_train_text[-1]:
                max_train_text = (train['id'], len(train['text'].split()))
            if len(train['title'].split()) > max_train_title[-1]:
                max_train_title = (train['id'], len(train['title'].split()))

        for _, test in test_df_all.iterrows():
            if len(test['text'].split()) > max_test_text[-1]:
                max_test_text = (test['id'], len(test['text'].split()))
            if len(test['title'].split()) > max_test_title[-1]:
                max_test_title = (test['id'], len(test['title'].split()))

        longest_train_title = train_df_all.loc[train_df_all['id'] == max_train_title[0]]['title']
        longest_train_text = train_df_all.loc[train_df_all['id'] == max_train_text[0]]['text']
        longest_test_title = test_df_all.loc[test_df_all['id'] == max_test_title[0]]['title']
        longest_test_text = test_df_all.loc[test_df_all['id'] == max_test_text[0]]['text']
        '''
        return train_titles, validation_titles, test_titles, train_texts, validation_texts, test_texts, train_labels, validation_labels, test_ids

