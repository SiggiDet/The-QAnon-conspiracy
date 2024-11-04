# The-QAnon-conspiracy


**Owners:**
*Óskar Guðmundur Kristinsson* & *Sigurður Detlef Jónsson*

**Course**
*REI505M*

## Project Description
Final Project In the Course REI505M

we work with data sets that were scraped from Reddit. The data contains posts to various 
subreddits (a subreddit is a community of users with an interest in a particular topic) 
before and after October 2017, the time when the first QAnon message was posted.

In this project we work with data sets that were scraped from Reddit. The data contains
posts to various subreddits (a subreddit is a community of users with an interest in a
particular topic) before and after October 2017, the time when the first QAnon message
was posted.
The first data set is from [1]. The file Hashed_Q_Submissions_Raw_Combined.csv
(900MB) contains anonymized posts of 2M submissions from 13.2K unique users who have
been identified as QAnon followers in the period October 2016 to January 2021. Reddit
banned 19 QAnon subbreddits in September 2018 which subsequently led to QAnon follow-
ers migrating to other platforms. This date and the affected subreddits served as reference
in identifying QAnon users (see [1] for details). The authors of [1] split the users into
QAnon-enthusiastic users, the most active posters, and QAnon-interested, the remaining
users. The definition of the two groups appears to be somewhat arbitrary and it is there-
fore of interest to study if it is strongly supported by the data. This can e.g., be done by
constructing a classifier to separate the two groups and analyzing the features most rele-
vant for classification. Each post has several fields, the message itself (text string), user
ID and metadata (number of upvotes, number of replies, date etc.) which could also be
useful. NMF could also give some insight into the data, e.g., what other topics the QAnon
2users are interested in (COVID-19, (anti-)vaccination etc.). This data set also includes
Hashed_allAuthorStatus.csv and Hashed_subredditStats.csv containing information
on the authors and subreddits, respectively. A description of the data and a link to it is
given here: https://github.com/sTechLab/QAnon_users_on_Reddit. Other possibilities
for analyzing this data set include tracking how the topics evolve over time.
To work with text in the algorithms that we’ve seen so far, the Reddit posts need to
be converted to vectorial form. Try something like CountVectorizer at first and logistic
regression or SVM. The next step would be to use pre-trained word embeddings like GloVe
or word2vec. Here each word is represented by a, say, 100-dimensional vector. A simple
way to convert a single post to vectorial format would be to average embedding vectors
corresponding to individual words in the post. After this, it would be interesting to try a
method that embed sentences directly, e.g., Sentence-BERT (https://www.sbert.net/).
Two Stanford students, Lillian Ma and Stephanie Vezich, carried out an interesting
study [2] where they used the above dataset as ”positive” examples, i.e., posts from QAnon
followers, and another dataset which they collected themselves, of non-QAnon followers, as
”negative” examples. This enabled them to train a classifier to predict whether a post is a
QAnon post or not which can subsequently be used to flag potentially harmful posts.
Ma and Vezich took this a step further. They were interested in whether it is possible
to identify posters that were susceptible to the QAnon conspiracy, before it came into
existence. That is do posts on Reddit prior to October 28th 2017 (the day of the first post
by ”Q”) by users which (later) turned out to be QAnon-enthusiastic or QAnon-interested,
provide any clues of their later ”conversion” to QAnon? Ma and Vezich used recurrent
neural networks in their study. Can you reproduce their results, to some extent, by using
a logistic regression classifier? The data set from can be downloaded from here: https:
//github.com/isvezich/cs230-political-extremism/tree/main
There are many possible avenues of exploration, including the study of differences in
phrasing between QAnon and non-QAnon users, classifying sentiment of QAnon and non-
QAnon posts (e.g., using a language model from Huggingface) and automatic removal of
bot posts during preprocessing. Does clustering provide insight into QAnon users? Do the
links in the QAnon posts provide any clues? The list goes on!

> Note: The datasets were originally obtained via an archiving service called
> PushShift by querying specific subreddits and users.This service is no longer
> publicly available but instead, a service called PullPush can be used
> (https://www.pullpush.io/).

## References
- [1] Engel et al. Characterizing Reddit Participation of Users Who Engage in the QAnon
Conspiracy Theories. Proc. ACM Hum.-Comput. Interact., Vol. 6, No. CSCW1, 53, 2022.
Link: https://dl.acm.org/doi/pdf/10.1145/3512900

- [2] Lillian Ma and Stephanie Vezich. Student project in CS230.
Link: http://cs230.stanford.edu/projects_fall_2022/reports/22.pdf
