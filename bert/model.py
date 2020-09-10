from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel
import torch
import torch.nn as nn
import numpy as np
import time


class BERTClassifier(nn.Module):
    def __init__(self, model_option, n_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_option)
        self.drop = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(self.bert.config.hidden_size * 2 + 11, 120)
        self.out = nn.Linear(self.bert.config.hidden_size * 2, n_classes)

    # consider metadata
    def forward(self, title_input_ids, text_input_ids, attention_mask1, attention_mask2):
        _, pooled_output1 = self.bert(
            input_ids=title_input_ids,
            attention_mask=attention_mask1
        )
        _, pooled_output2 = self.bert(
            input_ids=text_input_ids,
            attention_mask=attention_mask2
        )

        output = torch.cat((pooled_output1, pooled_output2), 1)
        # output = self.relu(self.fc1(output))
        output = self.drop(output)
        res = self.out(output)

        return res


class Model:

    @staticmethod
    def train_epoch(model, optimizer, scheduler, train_dataloader, n_examples, device, loss_fn):

        model.train()

        # Store the average loss after each epoch so we can plot them.
        losses = []
        correct_predictions = 0

        for data in train_dataloader:
            title_input_ids = data["title_input_ids"].to(device)
            text_input_ids = data["text_input_ids"].to(device)
            attention_mask1 = data["attention_mask1"].to(device)
            attention_mask2 = data["attention_mask2"].to(device)
            labels = data["labels"].to(device)

            outputs = model(
                title_input_ids=title_input_ids,
                text_input_ids=text_input_ids,
                attention_mask1=attention_mask1,
                attention_mask2=attention_mask2,
            )

            _, pred = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(pred == labels)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        return correct_predictions.double() / n_examples, np.mean(losses)

    @staticmethod
    def train_model(model, train_dataloader, validation_dataloader, train_len, validation_len, epochs, device, loss_fn):
        optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(epochs):

            # ========================================
            #               Training
            # ========================================

            print('')
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            # print(f'======== Epoch {epoch + 1} / {epochs} ========')
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            train_acc, train_loss = Model.train_epoch(model, optimizer, scheduler, train_dataloader, train_len, device,
                                                      loss_fn)

            print('Train loss: {:}, accuracy: {:}'.format(train_loss, train_acc))
            print('Epoch {:} took {:} minutes'.format(epoch + 1, (time.time() - t0) / 60))

            # ========================================
            #               Validation
            # ========================================

            print('')
            print("Running Validation...")

            val_acc, val_loss = Model.eval_model(model, validation_dataloader, validation_len, device, loss_fn)
            print('Validation loss: {:}, accuracy: {:}'.format(val_loss, val_acc))
            print('')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), 'best_model_state.bin')
                best_accuracy = val_acc

        print('')
        print('Total Training took: {:} minutes'.format((time.time() - total_t0) / 60))
        print('Best validation accuracy: {:}'.format(best_accuracy))
        return history

    @staticmethod
    def eval_model(model, validation_dataloader, n_examples, device, loss_fn):
        model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in validation_dataloader:
                title_input_ids = data["title_input_ids"].to(device)
                text_input_ids = data["text_input_ids"].to(device)
                attention_mask1 = data["attention_mask1"].to(device)
                attention_mask2 = data["attention_mask2"].to(device)
                labels = data["labels"].to(device)

                outputs = model(
                    title_input_ids=title_input_ids,
                    text_input_ids=text_input_ids,
                    attention_mask1=attention_mask1,
                    attention_mask2=attention_mask2,
                )

                _, pred = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)

                correct_predictions += torch.sum(pred == labels)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    @staticmethod
    def get_predictions(model, test_dataloader, device):
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []
        with torch.no_grad():
            # Predict
            for data in test_dataloader:
                title_input_ids = data["title_input_ids"].to(device)
                text_input_ids = data["text_input_ids"].to(device)
                attention_mask1 = data["attention_mask1"].to(device)
                attention_mask2 = data["attention_mask2"].to(device)

                outputs = model(
                    title_input_ids=title_input_ids,
                    text_input_ids=text_input_ids,
                    attention_mask1=attention_mask1,
                    attention_mask2=attention_mask2,
                )

                _, pred = torch.max(outputs, dim=1)

                # Store predictions and true labels
                predictions.extend(pred)

        predictions = torch.stack(predictions).cpu()
        return predictions
