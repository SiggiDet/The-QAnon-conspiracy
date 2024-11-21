import json
import os
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedKFold
import numpy as np

from model.utils import sentences_to_indices, read_glove_vecs
from model.read_data import prepare_average_word_embeddings, prepare_sequence_word_embeddings

def f1_m(recall,precision):
    return 2*((precision*recall)/(precision+recall+(0.000000001)))



def get_months_val(params,month_fpath):
    df = pd.read_csv(month_fpath)
    indices = np.random.choice(a=[0, 1], size=len(df), p=[.95, .05])
    df = df[indices == 1]
    words_test = tf.convert_to_tensor(df["words"], dtype=tf.string)
    labels_test = tf.convert_to_tensor(df["isUQ"])


    embeddings_path = './data/glove.6B.'+str(params.embedding_size)+'d.txt'
    if params.embeddings == 'GloVe':
        print("preparing word embeddings")
        if params.model_version == 'mlp':
            inputs = {
                'test': [words_test, labels_test],
                # just an empty sender does not use this variables for this case
                'train': [words_test,labels_test],
                'val': [words_test,labels_test]
            }
            inputs = prepare_average_word_embeddings(inputs, params, embeddings_path)
        else:
            inputs = {'test': [words_test, labels_test]}
            maxLen = params.max_word_length
            words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(embeddings_path)
            inputs['test'][0] = sentences_to_indices(inputs['test'][0], words_to_index, maxLen)
            inputs['test'].append(tf.data.Dataset.from_tensor_slices((inputs['test'][0], inputs['test'][1])) \
                .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
            inputs['word_to_vec_map'] = word_to_vec_map
            inputs['words_to_index'] = words_to_index
    else:
        inputs['test'].append(tf.data.Dataset.from_tensor_slices((words_test, labels_test)) \
            .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    return inputs['test'][0], inputs['test'][1], inputs['test'][2]

def train_and_evaluate_assist(params,model,train_ds,val_ds,test_ds,callbacks,class_weight):

    if params.model_version.startswith('BERT'):
        # Ragged tensors cannot be run on GPU
        with tf.device('/cpu:0'):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                callbacks=callbacks,
                epochs=params.num_epochs,
                class_weight=class_weight)
        loss, accuracy, precision, recall = model.evaluate(test_ds)

    else:
        with tf.device('/cpu:0'):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                callbacks=callbacks,
                class_weight=class_weight,
                epochs=params.num_epochs)
        loss, accuracy, precision, recall = model.evaluate(test_ds)
    
    return loss,accuracy,precision, recall, history

def retrieve_all_months(submissions_data_dir:str,curr_month=None):
    file_names = [f for f in os.listdir(submissions_data_dir) if os.path.isfile(os.path.join(submissions_data_dir, f))]

    if curr_month != None:
        file_names.remove(curr_month)

    return file_names
    
def perform_months_evaluation(params,model,train_ds,val_ds,test_ds,callbacks,class_weight,curr_month): 
    # create manually a list of month

    submissions_data_dir = "./data/months/"
    all_months = retrieve_all_months(submissions_data_dir,curr_month=curr_month)


    histories = []
    metrics = {
        "loss": [],
        "accuracy": [],
        "f1_m": [],
        "precision": [],
        "recall": [],
        "r_m": [],
        "p_m": [],
    }

    month_test_dataset = None # Initialising
    loss,accuracy,precision, recall, history =  train_and_evaluate_assist(params,model,train_ds,val_ds,test_ds,callbacks,class_weight)
    
    # iterate all months 
    for m in all_months:

        m_file_path = submissions_data_dir + m
        month_features_test, month_label_test, month_test_dataset = get_months_val(params,m_file_path)
        
        # Or model.evaluate
        loss, accuracy, precision, recall = history.evalute(month_test_dataset)
        metrics["loss"].append(loss)
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1_m(recall,precision))

        histories.append(history)

    
    return metrics, histories
    
    
def perform_cross_validation(features_train, labels_train, model, params,callbacks,class_weight):
 
    kfold = StratifiedKFold(n_splits=params.kfold, shuffle=True, random_state=42)

    histories = []
    init_weights = model.get_weights()

    fold_num = 1
    metrics = {
        "loss": [],
        "accuracy": [],
        "f1_m": [],
        "precision": [],
        "recall": [],
        "r_m": [],
        "p_m": [],
    }
    
    
    # Fit model from one month to thext 
    # create new model for each month 

    # for month in months, module do evaluate

    for train_idx, val_idx in kfold.split(features_train, labels_train):
        model.set_weights(init_weights)

        # Creating  training and validation sets with tf.gather
        X_train_fold = tf.gather(features_train, train_idx)
        X_val_fold = tf.gather(features_train, val_idx)
        y_train_fold = tf.gather(labels_train, train_idx)
        y_val_fold = tf.gather(labels_train, val_idx)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)).batch(params.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(params.batch_size)

        loss, accuracy, precision, recall, history = train_and_evaluate_assist(
            params, model, train_ds, val_ds, val_ds, callbacks,class_weight
        )

        histories.append(history)
        
        # Append metrics for later analysis
        metrics["loss"].append(loss)
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1_m(recall,precision))

        fold_num += 1

    return metrics, histories
    
def train_and_evaluate(inputs, model_path, model, params):
    """Evaluate the model

    Args:
        inputs: (dict) contains the inputs of the graph (features, labels...)
        model: (keras.Sequential) keras model with pre-defined layers
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
    """
    metrics = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    logdir = os.path.join(model_path+"/logs")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=params.early_stopping_patience),
        ModelCheckpoint(filepath=f"{model_path}/"
                                 f"best_model_"
                                 f"{params.model_version}_"
                                 f"embeddings:{params.embeddings}_"
                                 f"{params.l2_reg_lambda}_"
                                 f"{params.learning_rate}_"
                                 f"{params.batch_size}_"
                                 f"{params.dropout_rate}"
                                 f".keras", monitor='val_loss',
                        save_best_only=True),
        TensorBoard(logdir, histogram_freq=0)  # https://github.com/keras-team/keras/issues/15163
    ]

    features_train, labels_train, train_ds = inputs['train'][0], inputs['train'][1], inputs['train'][2]
    features_val, labels_val, val_ds = inputs['val'][0], inputs['val'][1], inputs['val'][2]
    features_test, labels_test, test_ds = inputs['test'][0], inputs['test'][1], inputs['test'][2]

    print(f"features_train shape: {features_train.shape}")
    print(f"labels_train shape: {labels_train.shape}")
    print(f"features_val shape: {features_val.shape}")
    print(f"labels_val shape: {labels_val.shape}")
    print(f"features_test shape: {features_test.shape}")
    print(f"labels_test shape: {labels_test.shape}")

    if params.class_weight_balance is False:
        class_weight = {0: 0.5, 1:0.5}
    else:
        class_weight = params.class_weight_balance

    if params.kfold != False: # Perform Cross Validation
        metrics,history = perform_cross_validation(features_train, labels_train, model, params,callbacks,class_weight)

    elif params.months_eval != False:
        print("running months evaluation")
        metrics,history = perform_months_evaluation(params,model,train_ds,val_ds,test_ds,callbacks,class_weight,curr_month=params.months_eval)

    else: 
        if params.model_version.startswith('BERT'):
            # Ragged tensors cannot be run on GPU
            with tf.device('/cpu:0'):
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    callbacks=callbacks,
                    epochs=params.num_epochs,
                    class_weight=class_weight)
            loss, accuracy, precision, recall = model.evaluate(test_ds)

        else:
            with tf.device('/cpu:0'):
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    callbacks=callbacks,
                    epochs=params.num_epochs,
                    class_weight=class_weight)
            loss, accuracy, precision, recall = model.evaluate(test_ds)

        metrics["loss"].append(loss)
        metrics["accuracy"].append(accuracy)
        metrics["f1"].append(f1_m(recall,precision))
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)

    for i in range(0,len(metrics['loss'])):
        test_history = {"loss": metrics["loss"][i], "binary_accuracy": metrics["accuracy"][i], "f1": metrics["f1"][i], "recall": metrics["recall"][i], "percision":metrics["precision"][i]}
   
        json.dump(test_history,
                  open(f"{model_path}/{i}"
                       f"test_history_model:{params.model_version}_"
                       f"embeddings:{params.embeddings}_"
                       f"h1units:{params.h1_units}_"
                       f"h2units:{params.h2_units}_"
                       f"l2reglambda:{params.l2_reg_lambda}_"
                       f"lr:{params.learning_rate}_"
                       f"batchsize:{params.batch_size}_"
                       f"dropout:{params.dropout_rate}.json", 'w'))

        print("Loss: ", metrics["loss"][i])
        print("accuracy: ", metrics["accuracy"][i])
        print("precision: ", metrics["precision"][i])
        print("recall: ", metrics["recall"][i])
        print("f1: ", metrics["f1"][i])

    return history
