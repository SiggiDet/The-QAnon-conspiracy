import pandas as pd
import os
import nltk

path = "../data/"

df_q_posts = pd.read_csv(os.path.join(path,'Users_isQ_words.csv'))
df_non_q_posts = pd.read_csv(os.path.join(path, 'non-q-posts-v2.csv'))
print(len(df_q_posts))
print(len(df_non_q_posts))

df_q_posts["isUQ"] = 1
#df_q_posts.assign(isUQ=1)
print(df_q_posts['isUQ'].value_counts())
def do_join(xs):
    if type(xs) == str or type(xs) == list:
        return " ".join([s for s in xs if type(s) == str])
    else:
        print(type(xs))
        return None

df_non_q_posts["title"] = df_non_q_posts["title"].astype(str)
df_non_q_posts["selftext"] = df_non_q_posts["selftext"].astype(str)
df_non_q_posts["words"] = (df_non_q_posts["title"] + df_non_q_posts["selftext"]).apply(do_join)
print(df_non_q_posts.head())
print(df_q_posts.head())
print(df_q_posts.columns)
print(df_non_q_posts.columns)
df_non_q_posts = df_non_q_posts.rename(columns={"hashed_author":"author","q_level":"isUQ","created_utc":"date_created"})
df_non_q_posts["isUQ"]=0
df = pd.concat([df_q_posts[['author','words','date_created','isUQ']], df_non_q_posts[['author','words','date_created','isUQ']]])
print(df['isUQ'].value_counts())
print(df["words"].head())
print(df.head())
df.dropna(subset=['author','words','isUQ'], inplace=True)
df["words"] = df["words"].astype(str)
df.to_csv('data/Users_isQ_words.csv', index=False)

print(df["isUQ"].value_counts())

import numpy as np

indices = np.random.choice(a=[0, 1], size=len(df), p=[.99, .01])
df[indices == 1].to_csv('data/Users_isQ_words_sanity.csv', index=False)
