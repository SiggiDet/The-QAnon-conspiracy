import argparse
import logging
import os
import sys

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.read_data import input_fn
from model.model_fn import model_fn
from model.visualize import metrics_to_plot


def check_path(path, file_usage='r'):
    """Check if a path exists and is accessible."""
    if not os.path.exists(path):
        print("Error! The path '{}' does not exist.".format(path))
        sys.exit(1)
    return True

def get_parser():
    """gathering parameters"""
    parser = argparse.ArgumentParser(description="Path to model_dir")
    
    # Define the argument
    parser.add_argument(
        '-p', '--path',  # Short and long version of the argument
        help="Path to parameters from the experiments params.json file in model_dir",
        required=True    # Make this argument mandatory
    )
    
    parser.add_argument(
        '-d', '--data',  # Short and long version of the argument
        help="Path to the data in csv format defaults to ./data/Users_isQ_words.csv",
        required=False    # Make this argument mandatory
    )

    # Parse the arguments
    args = parser.parse_args()
    
    # Check the validity of the provided path
    check_path(args.path)
    
    return args


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = get_parser()
    #model_dir = './expirements/mlp/'
    
    model_dir = args.path
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(model_dir,'train.log'))

    if args.data is not None:
        pos_dataset = args.data
    else:
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
    history = train_and_evaluate(inputs, model_dir, train_model, params)
    metrics_to_plot(history, params, model_dir)
