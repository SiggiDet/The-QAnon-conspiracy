import argparse
import logging
import os

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.read_data import input_fn
from model.model_fn import model_fn


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    json_path = os.path.join('./', 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger('train.log')

    pos_dataset = os.path.join('./data/', 'Users_isQ_words.csv')
    msg = "{} file not found. Make sure you have the right dataset"
    assert os.path.isfile(pos_dataset), msg.format(pos_dataset)

    logging.info("Creating the datasets...")
    print(f'Embeddings == {params.embeddings}: train')
    embeddings_path = None

    # Create the input tensors from the datasets
    inputs = input_fn(pos_dataset, params, embeddings_path)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model, inputs = model_fn(inputs, params)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    history = train_and_evaluate(inputs, train_model, params)
    metrics_to_plot(history, params)
