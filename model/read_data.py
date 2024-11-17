import pandas as pd
import os
import numpy as np
import tensorflow as tf

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
    

def input_fn(f_path, params, embeddings_path=None):
    """
    Args:
        f_path: (string) path to reddit posts
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        embeddings_path: (string) relative path to pre-trained embeddings if they are to be used, else None
    """
    # Load the dataset into mem
    df = pd.read_csv(f_path)

    # split into train/dev/test
    np.random.seed(0)
    indices = np.random.choice(a=[0, 1, 2], size=len(df), p=[.6, .2, .2])

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

    inputs = {
        'train': [words_train, labels_train],
        'val': [words_val, labels_val],
        'test': [words_test, labels_test],
    }

    print("Done reading in data")

    return inputs
