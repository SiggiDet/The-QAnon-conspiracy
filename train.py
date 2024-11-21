import argparse
import logging
import os
import sys
import keras
import tensorflow as tf
import keras_tuner
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from model.utils import Params
from model.utils import set_logger
from model.utils import add_hyper_param
from model.training import train_and_evaluate
from model.read_data import input_fn
from model.model_fn import model_fn
from model.visualize import metrics_to_plot
from model.hyper import tune_hyperparmeters


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
    #tf.config.threading.set_inter_op_parallelism_threads(112)
    tf.device('/cpu:0') #use all cores
    policy = keras.mixed_precision.Policy("float64")
    keras.mixed_precision.set_global_policy(policy)
    # Load the parameters from the experiment params.json file in model_dir
    args = get_parser()
    #model_dir = './expirements/mlp/'
    
    model_dir = args.path
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    print(params.early_stopping_patience)

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
    embeddings_path = './data/glove.6B.'+str(params.embedding_size)+'d.txt'

    # Create the input tensors from the datasets
    inputs = input_fn(pos_dataset, params, embeddings_path)
    logging.info("- done.")

    if not params.tune:
        # Define the models (2 different set of nodes that share weights for train and eval)
        logging.info("Creating the model...")
        train_model, inputs = model_fn(inputs, params)
        logging.info("- done.")

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        history = train_and_evaluate(inputs, model_dir, train_model, params)
        metrics_to_plot(history, params, model_dir)
    else:
        def hypermodel(hp):
            return tune_hyperparmeters(hp,params,inputs)
        tuner = keras_tuner.Hyperband(
            hypermodel=hypermodel,
            objective="val_binary_accuracy",
            executions_per_trial=3,
            overwrite=True, #becarefull
            directory=model_dir+"/hyper/",
            project_name="Qanon",
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=params.early_stopping_patience),
            TensorBoard(os.path.join(model_dir+"/hyper/logs"), histogram_freq=0)  # https://github.com/keras-team/keras/issues/15163
        ]
        if params.class_weight_balance is False:
            class_weight = {0: 0.5, 1:0.5}
        else:
            class_weight = params.class_weight_balance

        features_train, labels_train, train_ds = inputs['train'][0], inputs['train'][1], inputs['train'][2]
        features_val, labels_val, val_ds = inputs['val'][0], inputs['val'][1], inputs['val'][2]
        features_test, labels_test, test_ds = inputs['test'][0], inputs['test'][1], inputs['test'][2]
        tuner.search(train_ds,validation_data=val_ds,callbacks=callbacks,epochs=10,class_weight=class_weight)
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        for i in models:
            print(i.summary())
        print(best_model.summary())
        print(tuner.results_summary())


