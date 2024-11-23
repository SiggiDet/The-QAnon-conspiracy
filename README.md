# The-QAnon-conspiracy


**Owners:**
*Óskar Guðmundur Kristinsson* & *Sigurður Detlef Jónsson*

**Course**
*REI505M*

# Project Description 
This project is a part of the final project for the course Machine learning, REI505M. The aim of this project was to To understand how Qanon befell people. To do this we had to find a way to idendity accurately and quickly people who engage in Qanon. FOrtunatly Qanons are not shy about giving the researchers their data. Enabling researchesr to start to build models to classify wether a person is a Qanon or not based on their posts on social media. This report dicusses ways to claissfy a user as Qanon and examines the evolution of Qanon posts. 


# Datasets | QAnon_users_on_Reddit

For this project three large datasets were used. These datasets consist of data from a bunch of reddit posts that were posted by either QAnon followers or QAnon enthusiasts and were provided by the [1] research project Datasets for QAnon on Reddit.

- Hashed_Q_Comments_Raw_Combined:   contains about 10 million unique posts from October 28.2016 to January 23rd, 2021, and posted by 11 thousand users. 

- Hashed_Q_Submissions_Raw_Combined: consists of 2.1 million posts from October 28.2016 and 23rd of January 2021. These reddit posts are from 50 thousand subreddits and were posted by about 13.2 thousand users. 

- Hashed_allAuthorStatus: a dataset that consists of about 13.200 QAnon users.  

- non-q-posts-v2.csv: a dataset that consists of posts by 61000 reddit users that where not associated with Qanon 

> All data of the mentioned datasets contain hashed usernames for the sake of keeping the identity of the users private. 

A more in-detail description and a link to these datasets can be found in the following github repositories: https://github.com/sTechLab/QAnon_users_on_Reddit and https://github.com/isvezich/cs230-political-extremism.

Two datasets were created from the aforementioned datasets, **Users_isQ_words** and **Users_isQ_words_with_non_q**. The **Users_isQ_words** was constructed by combining the **Hashed_Q_Submissions_Raw_Combined** and **Hashed_Q_Comments_Raw_Combined datasets**. The Users_isQ_words dataset took the text and titles of posts and combined them to form the words column, from the submissions the only text from the user was already in a singular column, text, that was then also placed into the word's column. All rows with Nan values were then dropped. The datasets from Engels et al. unfortunately did not include posts that were not associated with Qanon, for that we had to turn to Ma & Vezich. They used a dataset of all reddit posts since 2006, publish on pushshift. Trouble with pushshift relegated us to using the dataset that they made available on their github containing posts, from non-Qanon users, from January 2017. This led to a wildly unbalanced dataset of 2692953 Qanon posts and 71818 non Qanon related posts.  

To combat the unbalanced dataset, we elected to try a few different approaches when training the models; oversampling the minority dataset, undersampling the majority dataset, and adding a class weight parameter to the loss function of the models.

## Requirements

### Data Dependencies
For the following project you'll need to create a data in the root location of the project directory containing the referred datasets mentioned in the [Datasets chapter](#datasets--qanon_users_on_reddit).

#### Pre-worded embeddings with GloVe
If you wish to use GloVe pre-worded embeddings you can do so by adding the pre-worded embeddings from GLoVe to a directory called ``correct_data/`` in the project repository and set in the ``params.json`` file that will be called with the train.py. Example of such use case can be found in the [Example UseCase chapter]().

## Dependencies
To setup the project you'll need conda and python.

### Set up Conda environment

To setup the environment that was used to run the project you'll need to follo
```bash
# 
cd The-QAnon-Conspiracy 

# Setup conda environment
conda env create -f environment.yml

# activate conda environment
conda activate tf

```

### Gathering data

Data needs to be added beforehand that is metnioned in the [Datasets chapter](#datasets--qanon_users_on_reddit) into the ``data`` directory at the root location on the repository


### Setup 

In the root directory run the following command
```bash
python3 setup.py
```

Next navigate to ``correct_data`` and run setup.py aswell
```bash
cd correct_data/
python3 setup.py
```

### Additional tools
Additional tools that where used in the project are located in the ``./tools/directory``.

## Data
All data that will be used should be moved into a directory called ``data/`` in the root of the repository. 

## Parameters
``train.py`` takes in two parameters, --path and --data.

- ``-p``: Takes in the location of the `params.json` file, do not specify that params.json just the location of the directory where it is contained. 
- ``-d``: Takes in the location of the dataset  to be used by the progarm.

### params.json
The user places it's parameters option that he wants to run the program with into the ``params.json`` file. An example of such use case can be found here:

#### Running with mlp
In the following example we provide a simple .json file that shows how to run the code with mlp

```json
{
    "model_version": "mlp",
    "max_features": 25000,
    "sequence_length": 1024,
    "h1_units": 256,
    "h2_units": 128,
    "l2_reg_lambda": 1e-3,
    "embedding_size": 512,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 2,
    "early_stopping_patience": 10,
    "dropout_rate": 0.0,
    "sample_rate": 1.0,
    "max_word_length": 60000,
    "h2_activation": "relu",
    "h1_activation": "relu",
    "embeddings": "None",
    "oversample": false,
    "undersample": false,
    "class_weight_balance": false,
    "output_activation": "sigmoid",
    "months_eval": false,
    "kfold": false,
    "tune": false,
    "loss": {
	"name": "BinaryCrossentropy",
	"from_logits": false,
	"label_smoothing": 0.0
    },
    "opt": {
	"name": "Adam",
        "clipnorm": 1.0,
	"beta_1": 0.9,
	"beta_2": 0.999,
	"amsgrad": false,
	"use_ema": false,
	"ema_momentum": 0.99,
	"ema_overwrite_frequency": "None"
    }
}

```

#### With log regression classifier with  GloVe 
In the following example we provide a simple .json file that shows how to run the code with logistic regression classifer with pre-trained embeddings from GloVe

```json
{
    "model_version": "log_reg",
    "max_features": 25000,
    "sequence_length": 1024,
    "h1_units": 256,
    "h2_units": 128,
    "l2_reg_lambda": 1e-3,
    "embedding_size": 512,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 2,
    "early_stopping_patience": 10,
    "dropout_rate": 0.0,
    "sample_rate": 1.0,
    "max_word_length": 60000,
    "h2_activation": "relu",
    "h1_activation": "relu",
    "embeddings": "None",
    "oversample": false,
    "undersample": false,
    "class_weight_balance": false,
    "output_activation": "sigmoid",
    "months_eval": false,
    "kfold": false,
    "tune": false,
    "loss": {
	"name": "BinaryCrossentropy",
	"from_logits": false,
	"label_smoothing": 0.0
    },
    "opt": {
	"name": "Adam",
        "clipnorm": 1.0,
	"beta_1": 0.9,
	"beta_2": 0.999,
	"amsgrad": false,
	"use_ema": false,
	"ema_momentum": 0.99,
	"ema_overwrite_frequency": "None"
    }
}
```

## Data preperation
Users_isQ_words 

### Simple Execution

To Run The code you simply need to type in the following command. 

```bash
python3 correct_data/train.py -p ./params_dir/ -d ./data/<path to data>
```

> Note! that it is recommended to have params.json file in it's own directory since the ouput will be located placed in the same location as the params.jso



## tools

Two tools where implemented

### split_data
By training a model on just one month of Qposts and then using the rest of the months as validation data, ie the month of 2017-11 will be evauated against 2017-12 then 2018-01 ...

To setup this data_set we use ``split_data.py``. 

```bash
python3 tools/split_data.py -p ./data/Hashed_Q_Submissions_Raw_Combined.csv -s ./data/months
```
> The csv file needs to have date_created section

Run month and then evalute it by then using the rest of the months as validation data this needs to be provided in the  ``params.json`` file like so:

```bash
    "months_eval": "2016_10",
```
> Here we want it to evaluate month 2016_10 and then using the rest of the months as validation data

To run this implementation type in the following at the root location of the repository:
```bash
python3 correct_data/train.py -p . -d ./data/months/2016_10.csv
```

````
## References

- [1] Lillian Ma and Stephanie Vezich. Student project in CS230.
Link: http://cs230.stanford.edu/projects_fall_2022/reports/22.pdf
