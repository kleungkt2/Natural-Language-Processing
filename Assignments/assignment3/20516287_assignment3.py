# Author: Leung Ko Tsun
# Student id: 20516287

import math
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

stopwords = set(stopwords.words("english"))
ps = PorterStemmer()
np.random.seed(0)


def load_data(file_name):
    """
    :param file_name: a file name, type: str
    return a list of ids, a list of reviews, a list of labels
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)

    return df['id'], df["text"], df['label']


def load_labels(file_name):
    """
    :param file_name: a file name, type: str
    return a list of labels
    """
    return pd.read_csv(file_name)['label']


def tokenize(text):
    """
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    e.g.
    Input: 'Text mining is to identify useful information.'
    Output: ['Text', 'mining', 'is', 'to', 'identify', 'useful', 'information', '.']
    """
    return nltk.word_tokenize(text)


def stem(tokens):
    """
    :param tokens: a list of tokens, type: list
    return a list of stemmed words, type: list
    e.g.
    Input: ['Text', 'mining', 'is', 'to', 'identify', 'useful', 'information', '.']
    Output: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.']
    """
    ### equivalent code
    # results = list()
    # for token in tokens:
    #     results.append(ps.stem(token))
    # return results

    return [ps.stem(token) for token in tokens]


def n_gram(tokens, n=1):
    """
    :param tokens: a list of tokens, type: list
    :param n: the corresponding n-gram, type: int
    return a list of n-gram tokens, type: list
    e.g.
    Input: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.'], 2
    Output: ['text mine', 'mine is', 'is to', 'to identifi', 'identifi use', 'use inform', 'inform .']
    """
    if n == 1:
        return tokens
    else:
        results = list()
        for i in range(len(tokens) - n + 1):
            # tokens[i:i+n] will return a sublist from i th to i+n th (i+n th is not included)
            results.append(" ".join(tokens[i:i + n]))
        return results


def filter_stopwords(tokens):
    """
    :param tokens: a list of tokens, type: list
    return a list of filtered tokens, type: list
    e.g.
    Input: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.']
    Output: ['text', 'mine', 'identifi', 'use', 'inform', '.']
    """
    ### equivalent code
    # results = list()
    # for token in tokens:
    #     if token not in stopwords and not token.isnumeric():
    #         results.append(token)
    # return results

    return [token for token in tokens if token not in stopwords and not token.isnumeric()]


def get_onehot_vector(feats, feats_dict):
    """
    :param data: a list of features, type: list
    :param feats_dict: a dict from features to indices, type: dict
    return a feature vector,
    """
    # initialize the vector as all zeros
    vector = np.zeros(len(feats_dict), dtype=np.float)
    for f in feats:
        # get the feature index, return -1 if the feature is not existed
        f_idx = feats_dict.get(f, -1)
        if f_idx != -1:
            # set the corresponding element as 1
            vector[f_idx] = 1
    return vector


def analyze_5_gram(train_stemmed, test_stemmed):
    # Create the list containing the 5_gram features from train corpus
    # Your Code Here
    train_5_gram = [n_gram(tokens, 5) for tokens in train_stemmed]

    # Create the list containing the 5_gram features from test corpus
    # Your Code Here
    test_5_gram = [n_gram(tokens, 5) for tokens in test_stemmed]

    # Build a Counter for 5-gram features
    # Your Code Here
    five_gram_feat_cnt = Counter()
    for feats in train_5_gram:
        five_gram_feat_cnt.update(feats)
    print("size of 5 gram features", len(five_gram_feat_cnt))

    # then, get the sorted features by the frequency

    five_gram_feat_keys = [f for f, cnt in five_gram_feat_cnt.most_common()]

    # draw linear lines and log lines for sorted features
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)

    # Your Code Here
    plt.plot(range(1, len(five_gram_feat_cnt)+1), [five_gram_feat_cnt[f] for f in five_gram_feat_keys])

    plt.xlabel("Feature Index")
    plt.ylabel("Feature Frequency")
    plt.subplot(1, 2, 2)

    # Your Code Here
    plt.loglog(range(1, len(five_gram_feat_cnt)+1),
               [five_gram_feat_cnt[f] for f in five_gram_feat_keys],
               basex=10, basey=10)

    plt.xlabel("Feature Index")
    plt.ylabel("Feature Frequency")

    print('Generating figure:')
    plt.savefig("5_gram.png")


def stem_three_gram_features(train_stemmed):
    # Create 3-gram feature list from the train set
    train_2_gram = [n_gram(tokens, 2) for tokens in train_stemmed]
    train_3_gram = [n_gram(tokens, 3) for tokens in train_stemmed]
    train_4_gram = [n_gram(tokens, 4) for tokens in train_stemmed]  # Your Code Here

    # build a set containing each unique feature which has appeared more than 10 times in the training set
    feats_set = set()

    # build a Counter for stemmed features, e.g., {"text": 2, "mine": 1}
    stemmed_feat_cnt = Counter()

    for feats in train_stemmed:
        stemmed_feat_cnt.update(feats)

    # add those stem features which occurs more than 10 times into the feature set.
    feats_set.update([f for f, cnt in stemmed_feat_cnt.items() if cnt > 10])

    # build a Counter for 3-gram features
    tri_gram_feat_cnt = Counter()

    for feats in train_3_gram:
        tri_gram_feat_cnt.update(feats)

    # add those 3-gram features which occurs more than 10 times into the feature set.
    feats_set.update([f for f, cnt in tri_gram_feat_cnt.items() if cnt > 10])

    print("Size of features:", len(feats_set))

    # build the feature dict mapping each feature to its index
    feats_dict = dict(zip(feats_set, range(len(feats_set))))  # Your Code Here

    # build the feature list
    train_feats = list()

    for i in range(len(train_ids)):
        # concatenate the stemmed token list and the 2_gram list together
        train_feats.append(train_stemmed[i] + train_2_gram[i])  # Your Code Here

    # build the feats_matrix
    # We first convert each example to a ont-hot vector, and then stack vectors as a matrix. Afterwards,
    # we save this feature matirx in a COO sparse matrix format to reduce memory consumption.

    # Your Code Here
    train_feats_matrix = np.vstack([get_onehot_vector(f, feats_dict) for f in train_feats])  # Your Code Here

    print('Number of stored values (including explicit zeros) in train_feats_matrix: ',
          train_feats_matrix.nonzero())

    return train_feats_matrix


def five_fold_cross_validation(train_feats_matrix, train_labels):
    # Define the number of folds and create the n-fold generator

    # Your Code Here
    n_fold = 5

    # create the n-fold generator
    # Your Code Here
    skf = StratifiedKFold(n_fold, shuffle=True)  # Your Code Here

    clfs = list()
    valid_acc_list = list()

    for k, (train_idx, valid_idx) in enumerate(skf.split(train_feats_matrix, train_labels)):  # Your Code Here
        # build the classifier and train
        clf = GaussianNB()

        # Your Code Here
        clf.fit(train_feats_matrix[train_idx], train_labels.values[train_idx])

        # Get the predictions of the classifier
        train_pred = clf.predict(train_feats_matrix[train_idx])
        valid_pred = clf.predict(train_feats_matrix[valid_idx])

        # Compute accuracy scores
        train_score = accuracy_score(train_labels.values[train_idx], train_pred)
        valid_score = accuracy_score(train_labels.values[valid_idx], valid_pred)

        print("training accuracy", train_score)
        print("validation accuracy", valid_score)

        clfs.append(clf)
        valid_acc_list.append(valid_score)

    print('Average validation score: ', sum(valid_acc_list) / len(valid_acc_list))


if __name__ == "__main__":
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    ans_file = "data/answer.csv"

    # load data
    train_ids, train_texts, train_labels = load_data(train_file)
    test_ids, test_texts, _ = load_data(test_file)
    test_labels = load_labels(ans_file)

    # extract features

    # tokenization
    train_tokens = [tokenize(text) for text in train_texts]
    test_tokens = [tokenize(text) for text in test_texts]

    # stemming
    train_stemmed = [stem(tokens) for tokens in train_tokens]
    test_stemmed = [stem(tokens) for tokens in test_tokens]

    train_stemmed = [filter_stopwords(tokens) for tokens in train_stemmed]
    test_stemmed = [filter_stopwords(tokens) for tokens in test_stemmed]

    print('Checking analyze_5_gram functions')

    analyze_5_gram(train_stemmed, test_stemmed)

    print('Checking stem_three_gram_features functions')

    train_feats_matrix = stem_three_gram_features(train_stemmed)

    print('Checking five_fold_cross_validation functions')

    five_fold_cross_validation(train_feats_matrix, train_labels)
