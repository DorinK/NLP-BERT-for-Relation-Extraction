from RelationExtraction import ProcessData, RelationExtractionDataset, TrainEvalModel, MODEL_FILE, format_time
from RelationExtraction import TRAIN_INPUT_FILE, TRAIN_GOLD_FILE, DEV_INPUT_FILE, DEV_GOLD_FILE
import sys
import os
import torch
import time

# Command-line arguments
INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]


def Train():

    start_time = time.time()

    # Process the data of the training and dev sets.
    print("Processing the data...")
    train = ProcessData(TRAIN_INPUT_FILE, TRAIN_GOLD_FILE)
    dev = ProcessData(DEV_INPUT_FILE, DEV_GOLD_FILE)
    print(f"Processing took {format_time(time.time() - start_time)} (hh:mm:ss)\n")

    # Create the DataLoaders for the training and dev sets.
    train_loader = RelationExtractionDataset(train.dataset)
    dev_loader = RelationExtractionDataset(dev.dataset)

    # Train the model
    trainer = TrainEvalModel()
    trainer.train(train_loader, train.gold, dev_loader, epochs=8)

    print(f"Overall, the whole run took {format_time(time.time() - start_time)} (hh:mm:ss)")


if __name__ == '__main__':

    # If the saved model is not in the current directory then train the model.
    if not os.path.exists(MODEL_FILE): Train()

    # Load the state dictionary of the model.
    state_dict = torch.load(MODEL_FILE)['model_state_dict']

    start_time = time.time()
    print('\nPredicting for test sentences...\n')

    # Process the test set data.
    test = ProcessData(INPUT_FILE)
    test_loader = RelationExtractionDataset(test.dataset)

    # Predict for the test sentences.
    TrainEvalModel(model_state_dict=state_dict).predict(test_loader, OUTPUT_FILE)

    print('DONE!')
    print(f"Predicting took {format_time(time.time() - start_time)} (hh:mm:ss)")
