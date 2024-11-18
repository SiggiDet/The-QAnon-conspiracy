import tensorflow as tf
from keras import layers
from keras.layers import Embedding, Input, Layer, TextVectorization
from keras.models import Model
import string
from nltk.corpus import stopwords
import re
import numpy as np

from keras import backend as K
from keras.losses import BinaryCrossentropy


def create_vectorized_layer(words, max_features):
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int')
    vectorize_layer.adapt(words)
    return vectorize_layer

# clean up junk from string
def custom_standardization(input_data):
    cachedStopWords = stopwords.words("english")

    lowercase = tf.strings.lower(input_data)
    for word in cachedStopWords:
        lowercase = tf.strings.regex_replace(lowercase, word, '')
    lowercase = tf.strings.regex_replace(lowercase, 'nan', '')
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

##
## TYPE ERRORS GALOR BELLOW
##
# define evaluation metrics                                                                                                                    
def recall_m(y_true, y_pred):                                                                                                                  
    y_pred = tf.math.sigmoid(y_pred)                                                                                                           
    true_positives = tf.math.cumsum(tf.math.round(tf.clip_by_value(tf.math.multiply(y_true, y_pred), 0, 1)))
    possible_positives = tf.math.cumsum(tf.math.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())                                                                               
    return recall

def precision_m(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    true_positives = tf.math.cumsum(tf.math.round(tf.clip_by_value(tf.math.multiply(y_true, y_pred), 0, 1)))
    predicted_positives = tf.math.cumsum(tf.math.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def word_mlp_model(params, vectorize_layer=None):
    print('no params.embeddings: model fn mlp model - 99')
    inputs = Input(shape=(), dtype='string')
    X_inp = vectorize_layer(inputs)
    X_inp = layers.Embedding(
        input_dim=len(vectorize_layer.get_vocabulary()),
        output_dim=params.embedding_size,
        # Use masking to handle the variable sequence lengths
        mask_zero=True)(X_inp)
    X_inp = layers.GlobalAveragePooling1D()(X_inp)
    X = layers.Dense(params.h1_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X_inp)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(params.h2_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X)
    X = layers.BatchNormalization()(X)
    outputs = layers.Dense(1, activation='sigmoid')(X)
    model = Model(inputs, outputs)
    return model

def model_fn(inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        inputs: features & labels
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (keras.Sequential) Sequential model
    """

    # set up model architecture
    if params.model_version == 'mlp':
        print('model version is mlp: model_fn - 159')
        vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
        model = word_mlp_model(params, vectorize_layer=vectorize_layer)
    elif params.model_version == 'rnn':
        vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
        model = word_rnn_model(params, vectorize_layer=vectorize_layer)
    elif params.model_version == 'lstm':
        vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
        model = word_lstm_model(params, vectorize_layer=vectorize_layer)
    elif params.model_version == 'BERT_LSTM':
        model = bert_to_lstm_model(params)
    elif params.model_version == 'BERT_RNN':
        model = bert_to_rnn_model(params)
    elif params.model_version == 'BERT_MLP':
        model = bert_to_mlp_model(params)
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    # compile model
    model.compile(loss=BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate, clipnorm=1.0),
                  metrics=[tf.metrics.BinaryAccuracy(),
                           f1_m,
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           ])
    print(model.summary())

    return model, inputs
