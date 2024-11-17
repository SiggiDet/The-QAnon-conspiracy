import pandas as pd
import os
import nltk
nltk.download('stopwords')

assert os.path.isfile('./data/Hashed_Q_Submissions_Raw_Combined.csv')
assert os.path.isfile('./data/Hashed_Q_Comments_Raw_Combined.csv')
assert os.path.isfile('./data/Hashed_allAuthorStatus.csv')

path = "./data"
df_posts = pd.read_csv(os.path.join(path,'Hashed_Q_Submissions_Raw_Combined.csv'))
df_comments = pd.read_csv(os.path.join(path,'Hashed_Q_Comments_Raw_Combined.csv'))
df_authors = pd.read_csv(os.path.join(path,'Hashed_allAuthorStatus.csv'))
df_posts.dropna(subset=['author'], inplace=True)
def do_join(xs):
    if type(xs) == str or type(xs) == list:
        return " ".join([s for s in xs if type(s) == str])
    else:
        return None

#this might work for posts but not for comments
df_posts["words"] = (df_posts["title"] + df_posts["text"]).apply(do_join)
#this works for comments
df_comments["words"] = df_comments["body"]
df = pd.concat([df_posts[['author','subreddit','words']], df_comments[['author','subreddit','words']]])
df = pd.merge(df, df_authors.rename(columns={"QAuthor": "author", "isUQ": "isUQ", "status":"status"}), on='author', how='outer')
df.dropna(subset=['words', 'isUQ'], inplace=True)
print(df['isUQ'].value_counts())
print(df["words"].head())
print(df.head())
df.to_csv('data/Users_isQ_words.csv', index=False)
