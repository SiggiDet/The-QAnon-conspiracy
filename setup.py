import pandas as pd
import os
import nltk
nltk.download('stopwords')

assert os.path.isfile('./data/Hashed_Q_Submissions_Raw_Combined.csv')
assert os.path.isfile('./data/Hashed_Q_Comments_Raw_Combined.csv')
assert os.path.isfile('./data/Hashed_allAuthorStatus.csv')

path = "./data"
df_posts = pd.read_csv(os.path.join(path,'Hashed_Q_Submissions_Raw_Combined.csv'))#,dtype={'subreddit':str,'id':str,'score':int,'numReplies':int,'author':str,'title':str,'text':str,'is_self':int,'domain':str,'url':str,'permalink':str,'upvote_ratio':float,'date_created':str})
df_comments = pd.read_csv(os.path.join(path,'Hashed_Q_Comments_Raw_Combined.csv'))#,dtype={'id':str,'link_id':str,'parent_id':str,'author':str,'subreddit':str,'body':str,'date_created':str})
df_authors = pd.read_csv(os.path.join(path,'Hashed_allAuthorStatus.csv'),dtype={'QAuthor':str,'isUQ':int,'status':str})
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
df = pd.concat([df_posts[['author','subreddit','words','date_created']], df_comments[['author','subreddit','words','date_created']]])
df = pd.merge(df, df_authors.rename(columns={"QAuthor": "author", "isUQ": "isUQ", "status":"status"}), on='author', how='outer')
df.dropna(subset=['words', 'isUQ'], inplace=True)
print(df['isUQ'].value_counts())
print(df["words"].head())
print(df.head())
df.dropna(subset=['author','words','isUQ'], inplace=True)
df["words"] = df["words"].astype(str)
df.to_csv('data/Users_isQ_words.csv', index=False)


