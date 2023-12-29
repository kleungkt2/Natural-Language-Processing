# author : Leung Ko Tsun
# studentid : 20516287
import numpy as np
from scipy import sparse
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

stopwords = set(stopwords.words("english"))

def tokenize(text):

    return nltk.word_tokenize(text)

def filter_stopwords(tokens):
  
    return [token for token in tokens if token not in stopwords and not token.isnumeric()]


def get_bagofwords(data, vocab_dict):

    data_matrix = sparse.lil_matrix((len(data), len(vocab_dict)))

    for i, doc in enumerate(data):
        for word in doc:
            word_idx = vocab_dict.get(word, -1)
            if word_idx != -1:
                data_matrix[i, word_idx] += 1
                
    data_matrix = data_matrix.tocsr()
    
    return data_matrix



def load_data(file_name):

    df = pd.read_csv(file_name)

    return df['id'], df["text"], df['label']

def load_labels(file_name):
 
    return pd.read_csv(file_name)['label']


def normalize(P, smoothing_prior=0):

    N = P.shape[0]
    
 
    norm = np.sum(P, axis=0, keepdims=True)
    

    return (P + smoothing_prior) / (norm + smoothing_prior*N)


def evaluate(y_true, y_pre):
    acc = accuracy_score(y_true, y_pre)
    return acc



def compute_prior(data_label, data_matrix):

    N = data_matrix.shape[0]
    K = max(data_label) # labels begin with 1

    # YOUR CODE HERE
    data_label_onehot_matrix = np.zeros((N, K))

    for i, l in enumerate(data_label):
        # YOUR CODE HERE
        data_label_onehot_matrix[i, l-1] = 1
    print('data_label_onehot_matrix.shape: ', data_label_onehot_matrix.shape)
    print('The label of the first three documents')
    print(data_label.values[:3])
    print()
    print('The first three rows of data_label_onehot_matrix')
    print(data_label_onehot_matrix[:3])
    # YOUR CODE HERE
    label_freq = np.sum(data_label_onehot_matrix, axis=0, keepdims=False)
    print("label_freq.shape:", label_freq.shape)
    print('Label\tFrequency')
    for l, f in enumerate(label_freq):
        print('{}\t{}'.format(l+1,f))
    
    # YOUR CODE HERE (use 1 as the smoothing prior)
    P_y = normalize(label_freq, smoothing_prior=1)
    print('P_y.shape: ', P_y.shape)
    print('Label\tPrior probability')
    for l, p in enumerate(P_y):
        print('{}\t{}'.format(l+1,p))
    print('train_data_matrix.shape: ', data_matrix.shape)#(N_train,V)
    print('train_data_matrix.transpose().shape: ', data_matrix.transpose().shape)#(V,N_train)
    print('data_label_onehot_matrix.shape: ', data_label_onehot_matrix.shape)#(N_train,K)
    return P_y, data_label_onehot_matrix



def compute_likelihood(data_matrix, data_label_onehot_matrix):

    # YOUR CODE HERE
    word_freq = train_data_matrix.transpose().dot(data_label_onehot_matrix)
    print("word_freq.shape:",word_freq.shape)
    # YOUR CODE HERE (use 1 as the smoothing prior)
    P_xy = normalize(word_freq,smoothing_prior=1)
    print('P_xy.shape', P_xy.shape)
    print('P_xy[:3, :]: ')
    P_xy[:3, :]
    return P_xy


def predict_NB(data_matrix, P_y, P_xy):

  # YOUR CODE HERE
    log_P_y = np.expand_dims(np.log(P_y), axis=0)
    print("log_P_y.shape: ",log_P_y.shape)
    log_P_xy = np.log(P_xy)
    print("log_P_xy.shape: ",log_P_xy.shape)
    log_P_dy = data_matrix.dot(log_P_xy)
    print("log_P_dy.shape: ", log_P_dy.shape)
    log_P = log_P_y + log_P_dy
    print("log_P.shape: ", log_P.shape)
    # YOUR CODE HERE
    pred = np.argmax(log_P, axis=1) + 1
    print("pred.shape: ", pred.shape)
    return pred




if __name__ == '__main__':
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    ans_file = "data/answer.csv"


    train_ids, train_texts, train_labels = load_data(train_file)
    test_ids, test_texts, _ = load_data(test_file)
    test_labels = load_labels(ans_file)

    print("Size of train set: {}".format(len(train_ids)))
    print("Size of test set: {}".format(len(test_ids)))


    train_tokens = [tokenize(text) for text in train_texts] 
    test_tokens = [tokenize(text) for text in test_texts]

    train_tokens = [filter_stopwords(tokens) for tokens in train_tokens]
    test_tokens = [filter_stopwords(tokens) for tokens in test_tokens]

    vocab = set()

    for i, doc in enumerate(train_tokens):
        for word in doc:
            vocab.add(word)
            
  
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    print('Size of vocab: ', len(vocab_dict))

    train_data_matrix = get_bagofwords(train_tokens, vocab_dict)
    test_data_matrix = get_bagofwords(test_tokens, vocab_dict)

    P_y, data_label_onehot_matrix = \
    compute_prior(train_labels, train_data_matrix)

    print('P_y.shape: ', P_y.shape)

    P_xy = compute_likelihood(train_data_matrix, data_label_onehot_matrix)

    print('P_xy.shape: ', P_xy.shape)

    train_pred = predict_NB(train_data_matrix, P_y, P_xy)
    
    test_pred = predict_NB(test_data_matrix, P_y, P_xy)


    train_acc= evaluate(train_labels, train_pred)
    print("Train Accuracy: {}".format(train_acc))

    test_acc= evaluate(test_labels, test_pred)
    print("Test Accuracy: {}".format(test_acc))
    print("Test Accuracy: {}".format(test_acc))
