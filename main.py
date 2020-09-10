import torch
from transformers import BertTokenizer
from data_processor.data_processor import DataProcessor
from bert.model import BERTClassifier, Model
import pandas as pd
from pathlib import Path


NUM_LABELS = 2
BATCH_SIZE = 16
EPOCHS = 1
MAX_LEN = 128


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

    train_titles, validation_titles, test_titles, train_texts, validation_texts, test_texts, train_labels, validation_labels, test_ids = DataProcessor.load_dataset()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_dataloader = DataProcessor.create_dataloader(train_titles, train_texts, train_labels, tokenizer, MAX_LEN, BATCH_SIZE)
    validation_dataloader = DataProcessor.create_dataloader(validation_titles, validation_texts, validation_labels, tokenizer, MAX_LEN, BATCH_SIZE)
    # kaggle test dataset doesn't have labels, set to -1, which is not used, that cannot evaluate the model accuracy on test dataset
    test_dataloader = DataProcessor.create_dataloader(test_titles, test_texts, [-1] * len(test_titles), tokenizer, MAX_LEN, BATCH_SIZE)

    device = get_device()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Train model

    model = BERTClassifier(model_option="bert-base-uncased", n_classes=NUM_LABELS)

    model = model.to(device)
    train_history = Model.train_model(model, train_dataloader, validation_dataloader, len(train_titles), len(validation_titles), EPOCHS, device, loss_fn)

    # predictions
    pred = Model.get_predictions(model, test_dataloader, device)
    predictions = {'id': test_ids, 'label': pred}
    prediction_df = pd.DataFrame(predictions, columns=['id', 'label'])
    RESULT_PATH = Path(__file__).parent / "data/result.csv"
    prediction_df.to_csv(RESULT_PATH, index=False, header=True)


if __name__ == "__main__":
    main()
