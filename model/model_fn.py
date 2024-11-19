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
    return tf.math.multiply(tf.cast(2.0,"float64"),tf.math.divide(tf.math.multiply(precision,recall),tf.math.add(precision,tf.math.add(recall,K.epsilon()))))

def oskarlogreg(params, vectorize_layer=None):
    print('no params.embeddings: model fn logreg model - 69420')
    inputs = Input(shape=(), dtype='string')
    vec_layer = vectorize_layer(inputs)
    Em = layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=params.embedding_size,mask_zero=True)(vec_layer)
    flat = layers.GlobalAveragePooling1D()(Em)
    outputs = layers.Dense(1, activation='sigmoid')(flat)
    model = Model(inputs, outputs)
    return model

def word_mlp_model(params, vectorize_layer=None, dropout=False):
    print('no params.embeddings: model fn mlp model - 99')
    inputs = Input(shape=(), dtype='string')
    X_inp = vectorize_layer(inputs)
    X_inp = layers.Embedding(
        input_dim=len(vectorize_layer.get_vocabulary()),
        output_dim=params.embedding_size,
        # Use masking to handle the variable sequence lengths
        mask_zero=False)(X_inp)
    X_inp = layers.GlobalAveragePooling1D()(X_inp)
    if dropout:
        X_inp = layers.Dropout(0.1)(X_inp)
    X = layers.Dense(params.h1_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X_inp)
    X = layers.BatchNormalization()(X)
    if dropout:
        X = layers.Dropout(0.1)(X)
    X = layers.Dense(params.h2_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X)
    X = layers.BatchNormalization()(X)
    if dropout:
        X = layers.Dropout(0.1)(X)
    outputs = layers.Dense(1, activation='sigmoid')(X)
    model = Model(inputs, outputs)
    return model

def setup_glove_embedding(params, glove_path,vectorize_layer=None):
    """
    Gather GloVe embeddings  and create an Matrix.
    Returns Matrix which is aligned with the vocabulary
    """

    embeddings_idx = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_idx[word] = coefs

    print("Found %s word vectors." % len(embeddings_idx))


# Example of preprocessinghttps://github.com/stanfordnlp/GloVe
def preprocess_text(text):
    return text.decode('utf-8')  # Ensure it's decoded correctly

def log_reg_classifier(params, vectorize_layer=None):
    """
    Implement a logistic regression classifier for data stored in Keras.
    """
    # Define the input layer for text data
    inputs = Input(shape=(), dtype='string', name="text_input")  # Input is raw text strings

    # Apply the vectorization layer
    X_inp = vectorize_layer(inputs)

    X_inp = layers.Embedding(
        input_dim=len(vectorize_layer.get_vocabulary()),
        output_dim=params.embedding_size,
        # Use masking to handle the variable sequence lengths
        mask_zero=True,
        name="embedding_layer"
        )(X_inp)

    # Reduce sequence dimension
    X_inp = layers.GlobalAveragePooling1D(name="pooling_layer")(X_inp)

    outputs = Dense(1, activation='sigmoid',name="output_layer")(X_inp)

    model = Model(inputs=inputs, outputs=outputs, name="log_Reg_model")

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
        model = word_mlp_model(params, vectorize_layer=vectorize_layer, dropout=params.dropout)
    elif params.model_version == 'oskarlogreg':
        vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
        model = oskarlogreg(params, vectorize_layer=vectorize_layer)
    elif params.model_version == 'rnn':
        vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
        model = word_rnn_model(params, vectorize_layer=vectorize_layer)
    elif params.model_version == 'lstm':
        vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
        model = word_lstm_model(params, vectorize_layer=vectorize_layer)
        #else:
        #    raise NotImplementedError("invalid embedding type")

    elif params.model_version == 'log_reg':
        print("creating log_reg classifier")

        # Implement Pre-trained word embeddings
        if params.embeddings == "GloVe":
            #params.embedding_size = 50
            params.embedding_size = 50

        # vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
        # model = log_reg_classifier(params,vectorize_layer=vectorize_layer)



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
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate), #, clipnorm=1.0),
                  metrics=[BinaryAccuracy(threshold=0.5, dtype=None),
                            f1_m,
                            keras.metrics.Precision(thresholds=0.5),
                            keras.metrics.Recall(thresholds=0.5),
                           ])
    print(model.summary())

    return model, inputs
