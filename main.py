import torch
import pickle
from transformers import BertTokenizer
from sklearn.metrics import classification_report
from data_processor.data_processor import DataProcessor
from bert.model import BERTClassifier, Model
from torch.utils.data import random_split


NUM_LABELS = 2
BATCH_SIZE = 16
EPOCHS = 4
MAX_LEN = 512


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        return device

    else:
        # print('No GPU available, using the CPU instead.')
        # device = torch.device("cpu")
        print('no GPU')
        exit()


def main():

    if NUM_LABELS == 2:
        class_names = ['True', 'Fake']
    else:
        class_names = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']

    train_titles, test_titles, train_texts, test_texts, train_labels, test_labels = DataProcessor.load_dataset()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataloader = DataProcessor.create_dataloader(train_titles, train_texts, train_labels, tokenizer, MAX_LEN, BATCH_SIZE)
    test_dataloader = DataProcessor.create_dataloader(test_titles, test_texts, test_labels, tokenizer, MAX_LEN, BATCH_SIZE)
    train_size = int(0.9 * len(train_titles))
    validation_size = len(train_titles) - train_size
    train_dataloader, validation_dataloader = random_split(dataloader, [train_size, validation_size], generator=torch.Generator().manual_seed(42))

    device = get_device()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Train model

    model = BERTClassifier(model_option="bert-base-uncased", n_classes=NUM_LABELS)

    model = model.to(device)
    train_history = Model.train_model(model, train_dataloader, validation_dataloader, train_size, validation_size, EPOCHS, device, loss_fn)

    # evaluate model on test dataset
    test_acc, _ = Model.eval_model(model, test_dataloader, len(test_titles), device, loss_fn)
    print('test accuracy: ', test_acc.item())

    # predictions
    pred, test_labels = Model.get_predictions(model, test_dataloader, device)

    with open('record.txt', 'wb') as f:
        pickle.dump(pred, f)
        pickle.dump(test_labels, f)

    print(classification_report(test_labels, pred, target_names=class_names))


if __name__ == "__main__":
    main()
