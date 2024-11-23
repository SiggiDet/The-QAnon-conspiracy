import tensorflow as tf
from keras import layers
from keras.layers import Embedding, Input, Layer, TextVectorization, Dense
from keras.models import Model, Sequential
import string
from nltk.corpus import stopwords
import re
import numpy as np
import keras

from keras import backend as K
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy

def create_vectorized_layer(words, max_features,out_mode='int'):
    """
    Creates and adapts a TextVectorization layer based on input words.
    """
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,  # Optional custom standardization
        max_tokens=max_features,             # Set the max vocabulary size
        output_mode=out_mode                 # Output as integer sequences (indices)
    )

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
    return_val = tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
    return return_val

def oskarlogreg(params, vectorize_layer=None):
    print('no params.embeddings: model fn logreg model - 69420')
    inputs = Input(shape=(), dtype='string')
    vec_layer = vectorize_layer(inputs)
    Em = layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=params.embedding_size,mask_zero=False)(vec_layer)
    flat = layers.GlobalAveragePooling1D()(Em)
    outputs = layers.Dense(1, activation='sigmoid')(flat)
    model = Model(inputs, outputs)
    return model

def word_mlp_model(params, vectorize_layer=None):
    if params.embeddings == 'GloVe':
        print('params.embeddings: model fn mlp model')
        inputs = Input(shape=(params.embedding_size,), dtype='float64')
        X_inp = inputs
    else:
        print('no params.embeddings: model fn mlp model - 99')
        inputs = Input(shape=(), dtype='string')
        X_inp = vectorize_layer(inputs)
        X_inp = layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=params.embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=False)(X_inp)
        X_inp = layers.GlobalAveragePooling1D()(X_inp)
    if params.dropout_rate > 0:
        X_inp = layers.Dropout(params.dropout_rate)(X_inp)
    X = layers.Dense(params.h1_units,
                           activation=params.h1_activation,
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X_inp)
    X = layers.BatchNormalization()(X)
    if params.dropout_rate > 0:
        X = layers.Dropout(params.dropout_rate)(X)
    X = layers.Dense(params.h2_units,
                           activation=params.h2_activation,
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X)
    X = layers.BatchNormalization()(X)
    if params.dropout_rate >0:
        X = layers.Dropout(params.dropout_rate)(X)
    outputs = layers.Dense(1, activation=params.output_activation)(X)
    model = Model(inputs, outputs)
    return model

# Example of preprocessinghttps://github.com/stanfordnlp/GloVe
def preprocess_text(text):
    return text.decode('utf-8')  # Ensure it's decoded correctly

def log_reg_classifier(params, vectorize_layer=None):
    """
    Implement a logistic regression classifier for data stored in Keras.
    """

    if vectorize_layer is None:
        
        # We replace float64 since GloVe replaces everything with float64
        inputs = Input(shape=(params.max_word_length,), dtype='float64', name="text_input")  # Input is raw text strings
        X_inp = inputs

    else:

        # Define the input layer for text data
        inputs = Input(shape=(), dtype='string', name="text_input")  # Input is raw text strings

        # Apply the vectorization layer
        X_inp = vectorize_layer(inputs)

        X_inp = layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=params.embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=False,
            name="embedding_layer"
            )(X_inp)

        # Reduce sequence dimension
        X_inp = layers.GlobalAveragePooling1D(name="pooling_layer")(X_inp)

    #X_inp = layers.Flatten()(X_inp)

    outputs = Dense(1, activation=params.output_activation,name="output_layer")(X_inp)

    model = Model(inputs=inputs, outputs=outputs, name="log_Reg_model")

    return model


def create_model(params, vectorize_layer=None):
    
    if vectorize_layer is None:
        inputs = Input(shape=(params.max_word_length,), dtype='float64', name="bagging")
        X_inp = inputs
    
    else:
        inputs = Input(shape=(), dtype='string', name="text_input")
        X_inp = vectorize_layer(inputs)

        X_inp = layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=params.embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=False,
            name="embedding_layer"
            )(X_inp)


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
        if params.embeddings == 'GloVe':
            params.embedding_size = 50
            model = word_mlp_model(params)
        else:
            print('model version is mlp: model_fn - 159')
            vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
            model = word_mlp_model(params, vectorize_layer=vectorize_layer)

    elif params.model_version == 'log_reg':
        print("creating log_reg classifier")
    
        # Implement Pre-trained word embeddings
        if params.embeddings == "GloVe":
            print("executing GloVe")
            model = log_reg_classifier(params)
        else:
            vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
            model = log_reg_classifier(params,vectorize_layer=vectorize_layer)

    loss = None
    if params.loss["name"] == "BinaryCrossentropy":
        loss = BinaryCrossentropy(from_logits=params.loss["from_logits"],label_smoothing=params.loss["label_smoothing"])
    elif params.loss["name"] == "BinaryFocalCrossentropy":
        if not params.class_weight_balance:
            loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False,alpha=0.0,gamma=params.loss["gamma"],from_logits=params.loss["from_logits"],label_smoothing=params.loss["label_smoothing"])
        else:
            loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=params.class_weight_balance[1],gamma=params.loss["gamma"],from_logits=params.loss["from_logits"],label_smoothing=params.loss["label_smoothing"])
    elif params.loss["name"] == "Poisson":
        loss = tf.keras.losses.Poisson()
    elif params.loss["name"] == "Dice":
        loss = tf.keras.losses.Dice()
    elif params.loss["name"] == "CosineSimilarity":
        loss = tf.keras.losses.CosineSimilarity()
    elif params.loss["name"] == "MSE":
        loss = tf.keras.losses.MSE()
    elif params.loss["name"] == "Tversky":
        loss = tf.keras.losses.Tversky(alpa=params.loss["alpha"],beta=params.loss["beta"])

    opt = None
    if params.opt["name"] == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=params.learning_rate, clipnorm=params.opt["clipnorm"], beta_1=params.opt["beta_1"], beta_2=params.opt["beta_2"], amsgrad=params.opt["amsgrad"], use_ema=params.opt["use_ema"],ema_momentum=params.opt["ema_momentum"],ema_overwrite_frequency=params.opt["ema_overwrite_frequency"])
    elif params.opt["name"] == "SGD":
        opt = keras.optimizers.SGD(learning_rate=params.learning_rate, momentum=params.opt["momentum"], nesterov=params.opt["nesterov"], weight_decay=params.opt["weight_decay"], clipnorm=params.opt["clipnorm"],use_ema=params.opt["use_ema"],ema_momentum=params.opt["ema_momentum"],ema_overwrite_frequency=params.opt["ema_overwrite_frequency"])
    elif params.opt["name"] == "AdamW":
        opt = tf.keras.optimizers.AdamW(learning_rate=params.learning_rate, clipnorm=params.opt["clipnorm"], rho=params.opt["rho"], amsgrad=params.opt["amsgrad"], use_ema=params.opt["use_ema"],ema_momentum=params.opt["ema_momentum"],ema_overwrite_frequency=params.opt["ema_overwrite_frequency"])
    elif params.opt["name"] == "Nadam":
        opt = tf.keras.optimizers.Adam(learning_rate=params.learning_rate, clipnorm=params.opt["clipnorm"], beta_1=params.opt["beta_1"], beta_2=params.opt["beta_2"], use_ema=params.opt["use_ema"],ema_momentum=params.opt["ema_momentum"],ema_overwrite_frequency=params.opt["ema_overwrite_frequency"])

    # compile model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[BinaryAccuracy(threshold=0.5, dtype=None),
                            keras.metrics.Precision(thresholds=0.5),
                            keras.metrics.Recall(thresholds=0.5),
                           ])
    print(model.summary())

    return model, inputs
