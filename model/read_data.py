import pandas as pd
import os
import numpy as np
import tensorflow as tf

from model.utils import read_glove_vecs, sentence_to_avg, sentences_to_indices

def load_data_to_df(path,zipped=False):
    ##
    ##  ==> data/Hashed_Q_Submissions_Raw_Combined.csv <==
    ##  subreddit,id,score,numReplies,author,title,text,is_self,domain,url,permalink,upvote_ratio,date_created
    ##  ==> data/Hashed_Q_Comments_Raw_Combined.csv <==
    ##  id,link_id,parent_id,author,subreddit,body,date_created
    ##
    if zipped:
        df_posts = pd.read_csv(os.path.join(path,'hashed_q_submissions_raw_combined.csv.gz'), compression='gzip')
        df_comments = pd.read_csv(os.path.join(path,'hashed_q_comments_raw_combined.csv.gz'), compression='gzip')
        df_authors = pd.read_csv(os.path.join(path,'hashed_allauthorstatus.csv.gz'), compression='gzip')
    else:
        df_posts = pd.read_csv(path)
        df_comments = pd.read_csv(path)
        df_authors = pd.read_csv(path)
    df_posts.dropna(subset=['author'], inplace=True)
    def do_join(xs):
        if type(xs) == str or type(xs) == list:
            return " ".join([s for s in xs if type(s) == str])
        else:
            return None

    df_posts["words"] = (df_posts["title"] + df_posts["text"]).apply(do_join)
    df_comments["words"] = df_comments["body"]
    df = pd.concat([df_posts[['author','subreddit','words']], df_comments[['author','subreddit','words']]])
    df = pd.merge(df, df_authors.rename(columns={"QAuthor": "author", "isUQ": "isUQ", "status":"status"}), on='author', how='outer')
    df.dropna(subset=['words', 'isUQ'], inplace=True)
    print(df['isUQ'].value_counts())
    print(df["words"].head())
    print(df.head())

    return df
    
def prepare_sequence_word_embeddings(inputs, params, embeddings_path):
    maxLen = params.max_word_length
    words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(embeddings_path)
    inputs['train'][0] = sentences_to_indices(inputs['train'][0], words_to_index, maxLen)
    print('finished sentences_to_indices for train')
    inputs['val'][0] = sentences_to_indices(inputs['val'][0], words_to_index, maxLen)
    print('finished sentences_to_indices for val')
    inputs['test'][0] = sentences_to_indices(inputs['test'][0], words_to_index, maxLen)
    print('finished sentences_to_indices for test')
    inputs['train'].append(tf.data.Dataset.from_tensor_slices((inputs['train'][0], inputs['train'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['val'].append(tf.data.Dataset.from_tensor_slices((inputs['val'][0], inputs['val'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['test'].append(tf.data.Dataset.from_tensor_slices((inputs['test'][0], inputs['test'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['word_to_vec_map'] = word_to_vec_map
    inputs['words_to_index'] = words_to_index
    return inputs

def prepare_average_word_embeddings(inputs, params, embeddings_path):
    words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(embeddings_path)
    # Get a valid word contained in the word_to_vec_map.
    str_feat_train = []
    str_feat_val = []
    str_feat_test = []
    # inputs['train'][0] is words_train, a 1D string tensor, 1 string per author / label
    # inputs['train'][0].shape[0] is (n,) for dataset with n examples
    for i in range(inputs['train'][0].shape[0]):  # for each example:
        author_text = inputs['train'][0][i]
        str_feat_train.append(sentence_to_avg(author_text, word_to_vec_map))
    print('finished sentence_to_avg for train')
    for i in range(inputs['val'][0].shape[0]):
        str_feat_val.append(sentence_to_avg(inputs['val'][0][i], word_to_vec_map))
    print('finished sentence_to_avg for val')
    for i in range(inputs['test'][0].shape[0]):
        str_feat_test.append(sentence_to_avg(inputs['test'][0][i], word_to_vec_map))
    print('finished sentence_to_avg for test')
    inputs['train'][0] = tf.cast(tf.stack(str_feat_train), 'float64')
    inputs['val'][0] = tf.cast(tf.stack(str_feat_val), 'float64')
    inputs['test'][0] = tf.cast(tf.stack(str_feat_test), 'float64')
    inputs['train'].append(tf.data.Dataset.from_tensor_slices((inputs['train'][0], inputs['train'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['val'].append(tf.data.Dataset.from_tensor_slices((inputs['val'][0], inputs['val'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['test'].append(tf.data.Dataset.from_tensor_slices((inputs['test'][0], inputs['test'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['word_to_vec_map'] = word_to_vec_map
    inputs['words_to_index'] = words_to_index
    return inputs

def input_fn(f_path, params, embeddings_path=None):
    """
    Args:
        f_path: (string) path to reddit posts
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        embeddings_path: (string) relative path to pre-trained embeddings if they are to be used, else None
    """
    # Load the dataset into mem
    df = pd.read_csv(f_path)

    if params.oversample or params.undersample:
        minor = df[df['isUQ'] == 1]
        major = df[df['isUQ'] == 0]
        len_ma = len(major)
        len_mi = len(minor)

        if params.oversample:
            minor = minor.sample(n=len_ma, replace=True, random_state = 10)
        if params.undersample:
            major = major.sample(n=len_mi, replace=True, random_state = 10) 
        df = pd.concat([major, minor])
        # shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)

    if params.class_weight_balance:
        params.class_weight_balance = {0: len(df[df['isUQ'] == 0]),1: len(df[df['isUQ'] == 1])}

    # split into train/dev/test
    np.random.seed(0)
    indices = np.random.choice(a=[0, 1, 2, 3], size=len(df), p=[.36, .12, .12, .6])

    print("Splitting data into train dev test")
    train_df = df[indices == 0]
    val_df = df[indices == 1]
    test_df = df[indices == 2]

    words_train = tf.convert_to_tensor(train_df["words"], dtype=tf.string)
    words_val = tf.convert_to_tensor(val_df["words"], dtype=tf.string)
    words_test = tf.convert_to_tensor(test_df["words"], dtype=tf.string)

    labels_train = tf.convert_to_tensor(train_df["isUQ"])
    labels_val = tf.convert_to_tensor(val_df["isUQ"])
    labels_test = tf.convert_to_tensor(test_df["isUQ"])

    print(val_df["isUQ"].value_counts())
    print(train_df["isUQ"].value_counts())
    print(test_df["isUQ"].value_counts())

    inputs = {
        'train': [words_train, labels_train],
        'val': [words_val, labels_val],
        'test': [words_test, labels_test],
    }

    if params.embeddings == 'GloVe':
        print("preparing word embeddings")
        if params.model_version == 'mlp':
            inputs = prepare_average_word_embeddings(inputs, params, embeddings_path)
        else:
            inputs = prepare_sequence_word_embeddings(inputs, params, embeddings_path)
    else:
        inputs['train'].append(tf.data.Dataset.from_tensor_slices((words_train, labels_train)) \
            .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
        inputs['val'].append(tf.data.Dataset.from_tensor_slices((words_val, labels_val)) \
            .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
        inputs['test'].append(tf.data.Dataset.from_tensor_slices((words_test, labels_test)) \
            .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))

    print("Done reading in data")

    return inputs
