#%%
#!/usr/bin/python
# essential imports
import warnings  # Ignore warnings

warnings.filterwarnings("ignore")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re  # regular expressions
import math  # math functions
import random  # random numbers and generator
import copy  # copy objects
import pickle  # copy objects into binary files
import timeit  # timer
import os  # system functions
import sys

import matplotlib.pyplot as plt  # plotting tool

#%matplotlib inline
plt.ion()

# scikit-learn
# evaluation metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

# preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# preprocess text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import cleanUp

# pytorch
import torch

# tensorflow
# import tensorflow as tf
# print(tf.__version__)
# pd options
# pd.set_option("display.max_columns", 500)
# pd.set_option("display.max_rows", 500)
# pd.set_option("display.width", 1000)

#%%
# default seeding for reproducability
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)


#%%
# Load civil comments datasets into pandas data frame
try:
    cwd = os.path.dirname(os.path.abspath(__file__))
except:
    cwd = os.getcwd()

# sample_path = os.path.join(cwd, "data", "sample_submission.csv")
# test_path = os.path.join(cwd, "data", "test.csv")
train_path = os.path.join(cwd, "data", "train.csv")


# sample_submission = pd.read_csv(sample_path) #sample submission format with id and prediction
# test_comments = pd.read_csv(test_path) #test comments with id and comment
train_comments = pd.read_csv(train_path)  # train comments with multiple attributes

#%%
# display head
train_comments.head()
# display first comment
train_comments.iloc[0]["comment_text"]
# display toxic comments above target 0.5
train_comments[train_comments["target"] >= 0.5].head()
# shuffle
#train_comments = train_comments.sample(frac=1).reset_index(drop=True)

# comment-target split
full_labels = train_comments.iloc[:]["target"].copy()
full_comments = train_comments[["comment_text"]].copy()
print(full_labels.head())
print(full_comments.head())


# %%
# split train into training-evaluation set 80%-20%
x_train, x_eval, y_train, y_eval = train_test_split(
    full_comments, full_labels, test_size=0.2, random_state=42, shuffle=False
)
# array form
# x_train.values
# %%
# TODO test preprocessing
#x_train.loc[:, "comment_text"] = x_train.comment_text.apply(cleanUp)
#x_val.loc[:, "comment_text"] = x_val.comment_text.apply(cleanUp)

