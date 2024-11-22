import argparse
import pandas as pd
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

from model.utils import Params

def get_parser():
    """gathering parameters"""
    parser = argparse.ArgumentParser(description="Path to model_dir")
    
    # Define the argument
    parser.add_argument(
        '-p', '--path',  # Short and long version of the argument
        help="Path to the .json file",
        required=True    # Make this argument mandatory
    )
    
    parser.add_argument(
        '-k', '--keras_m',  # Short and long version of the argument
        help="Path to the .keras file",
        required=True    # Make this argument mandatory
    )

    # Parse the arguments
    args = parser.parse_args()
    
    # Check the validity of the provided path
    
    return args
@keras.saving.register_keras_serializable()
def f1_m(y_true, y_pred):
    return y_true
@keras.saving.register_keras_serializable()
def precision_m(y_true, y_pred):
    return y_true
@keras.saving.register_keras_serializable()
def recall_m(y_true, y_pred):
    return y_true

if __name__ == '__main__':
    #tf.config.threading.set_inter_op_parallelism_threads(112)
    tf.device('/cpu:0') #use all cores
    policy = keras.mixed_precision.Policy("float64")
    keras.mixed_precision.set_global_policy(policy)
    # Load the parameters from the experiment params.json file in model_dir
    args = get_parser()
    #model_dir = './expirements/mlp/'
    
    keras_path = args.keras_m
    json_path = args.path
    params = Params(os.path.join(json_path, 'params.json'))
    dataset = os.path.join('./data/', 'Users_isQ_words.csv')
    df = pd.read_csv(dataset)
    words = tf.convert_to_tensor(df["words"], dtype=tf.string)
    labels = tf.convert_to_tensor(df["isUQ"])
    test = tf.data.Dataset.from_tensor_slices((words, labels)).shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size)
    model = tf.keras.models.load_model(keras_path)
    loss,accuracy,f1_m,p_m,r_m,precision, recall = model.evaluate(test)
    print(loss)
    print(accuracy)
    print(precision)
    print(recall)

