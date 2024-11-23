import json
import os
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import KFold

def train_and_evaluate_assist(params,model,train_ds,val_ds,test_ds,callbacks):

    if params.model_version.startswith('BERT'):
        # Ragged tensors cannot be run on GPU
        with tf.device('/cpu:0'):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                callbacks=callbacks,
                epochs=params.num_epochs)
        loss, accuracy, f1, r_m, p_m, precision, recall = model.evaluate(test_ds)

    else:
        with tf.device('/cpu:0'):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                callbacks=callbacks,
                epochs=params.num_epochs)
        loss, accuracy, f1, r_m, p_m, precision, recall = model.evaluate(test_ds)
    
    return loss,accuracy,f1,r_m,p_m,precision, recall, history


def perform_cross_validation(features_train, labels_train, model, params,callbacks,metrics):
 
    kfold = KFold(n_splits=params.kfold, shuffle=True, random_state=42)

    histories = []
    init_weights = model.get_weights()

    fold_num = 1

    for train_idx, val_idx in kfold.split(range(features_train.shape[0])):

        model.set_weights(init_weights)

        # splitting the data twith tf.gather to train and val
        X_train_fold = tf.gather(features_train, train_idx)
        X_val_fold = tf.gather(features_train, val_idx)
        y_train_fold = tf.gather(labels_train, train_idx)
        y_val_fold = tf.gather(labels_train, val_idx)

        # re-creating  objects of type tf.data.Datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)).batch(params.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(params.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(params.batch_size)

        loss, accuracy, f1, r_m, p_m, precision, recall,history = train_and_evaluate_assist(
            params, model, train_ds, val_ds, test_ds, callbacks
        )

        histories.append(history)
        # Log metrics for the current fold
        print(f"Fold {fold_num} -- Loss: {loss}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        metrics["loss"].append(loss)
        metrics["accuracy"].append(accuracy)
        metrics["f1_m"].append(f1)
        metrics["r_m"].append(r_m)
        metrics["p_m"].append(p_m)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)

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
        "f1_m": [],
        "precision": [],
        "recall": [],
        "r_m": [],
        "p_m": [],
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


    if params.kfold != False: # Perform Cross Validation
        metrics,history = perform_cross_validation(features_train, labels_train, model, params,callbacks,metrics)
        loss, accuracy, f1_m, r_m, p_m, precision, recall = metrics   
        metrics_iter = params.kfold
    else: 
        if params.model_version.startswith('BERT'):
            # Ragged tensors cannot be run on GPU
            with tf.device('/cpu:0'):
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    callbacks=callbacks,
                    epochs=params.num_epochs)
            loss, accuracy, f1_m, r_m, p_m, precision, recall = model.evaluate(test_ds)
            metrics["loss"].append(loss)
            metrics["accuracy"].append(accuracy)
            metrics["f1_m"].append(f1)
            metrics["r_m"].append(r_m)
            metrics["p_m"].append(p_m)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
        else:
            with tf.device('/cpu:0'):
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    callbacks=callbacks,
                    epochs=params.num_epochs)
            loss, accuracy, f1_m, r_m, p_m, precision, recall = model.evaluate(test_ds)
            metrics["loss"].append(loss)
            metrics["accuracy"].append(accuracy)
            metrics["f1_m"].append(f1)
            metrics["r_m"].append(r_m)
            metrics["p_m"].append(p_m)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            #loss, accuracy, f1_m, precision, recall = model.evaluate(test_ds)
    
    for i in range(0,len(metrics["p_m"])):
       # test_history = {"loss": loss, "binary_accuracy": accuracy, "f1_m": f1, "precision_m": precision, "recall_m": recall}
        test_history = {"loss": metrics["loss"][i], "binary_accuracy": metrics["accuracy"][i], "f1_m": metrics["f1_m"][i], "r_m": metrics["r_m"[i], "p_m":metrics["p_m"][i], "recall": metrics["recall"][i], "percision":metrics["precision"[i]}
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
        print("f1_m: ", metrics["f1_m"][i])
    #     print("AUC: ", auc)

    return history
