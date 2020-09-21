"""
Naive Bayes: Movie Review Prediction
Last update: KzXuan, 2020.04.10
"""
import os
import re
import math
import itertools
import numpy as np
import codecs as cs
import time

# get stop words list
with cs.open("stop_words_zh.txt",'r',encoding='utf-8') as fobj:
    stop_words = [line.rstrip() for line in fobj.readlines()]


def read_file(fname, encoding='utf-8', sample_tag='text'):
    """
    Read file with sample tag like <text>...</text>.
    Improved from PyTC.read_file_f2() from Rui Xia.

    Args:
        encoding [str]: encoding standard
        sample_tag [str]: the separator between samples

    Returns:
        res [list]: all the texts
    """
    with cs.open(fname, 'r', encoding=encoding, errors='ignore') as fobj:
        train_positive = fobj.read()
        patn = r'<{}>([\s\S]*?)</{}>'.format(sample_tag, sample_tag)
        res = re.findall(patn, train_positive)
        res = [list(filter(lambda x: x not in stop_words, r.split())) for r in res]
        # res = [r.split() for r in res]
    return res


def load_data(path):
    """
    Load data from a determined path.

    Args:
        path [str]: file path

    Returns:
        data [list]: all the text data
        labels [list]: all the labels
    """
    data, labels = [], []
    # get files under the path
    files = [f for f in os.listdir(path) if f[0] != '.']
    for i, f in enumerate(files):
        res = read_file(path + f)
        data += res
        labels += [i] * len(res)
    return data, labels


class NaiveBayes(object):
    def __init__(self, train_data, train_labels):
        """
        Naive bayes.

        Args:
            train_data [list]: each element is a document, like [['I', 'am', 'happy', '.'],]
            train_labels [list]: train data labels, like [1, 0, 1, ...] (len(train_labels) == len(train_data))
        """
        self.train_data = train_data
        self.train_labels = train_labels
        # class/category number
        self.n_class = max(train_labels) + 1
        # class probability
        self.class_probability = [self.train_labels.count(c) / len(self.train_labels) for c in range(self.n_class)]
        # unigram word list
        self.word_list = self.unigram_list()
        self.n_word = len(self.word_list)

    def unigram_list(self):
        """
        Get unigram word list.

        Returns:
            uni_l [set]: all the words
        """
        uni_l = set(itertools.chain(*self.train_data))
        return uni_l

    def multinomial(self, test_data, test_labels):
        """
        Multinomial naive bayes.

        Args:
            test_data [list]: each element is a document
            test_labels [list]: test data labels
        """
        # calculate term frequency
        self.frequency = [{}.fromkeys(self.word_list, 0) for _ in range(self.n_class)]
        for i, doc in enumerate(self.train_data):
            label = self.train_labels[i]
            for word in doc:
                self.frequency[label][word] += 1

        # calculate probability
        self.probability = []
        for f in self.frequency:
            _sum = sum(f.values()) + len(f)
            tp = dict(zip(f.keys(), [(v + 1.) / _sum for v in f.values()]))
            self.probability.append(tp)

        # do test prediction
        test_prediction = []
        for data in test_data:
            preds = []
            for i, p in enumerate(self.probability):
                # add log value instead of multiplication
                pred = math.log(self.class_probability[i])
                for word in data:
                    if word in p:
                        pred += math.log(p[word])
                preds.append(pred)
            # do argmax to get prediction label
            preds = np.argmax(preds)
            test_prediction.append(preds)
        # calculate accuracy
        accuracy = np.sum(np.array(test_prediction) == np.array(test_labels)) / len(test_labels)
        return accuracy

    def bernoulli(self, test_data, test_labels, alpha=0.1):
        """
        Multi-variate bernoulli naive bayes.

        Args:
            test_data [list]: each element is a document
            test_labels [list]: test data labels
            alpha [float]: smooth parameter
        """
        # calculate document frequency
        self.frequency = [{}.fromkeys(self.word_list, 0) for _ in range(self.n_class)]
        for i, doc in enumerate(self.train_data):
            label = self.train_labels[i]
            doc = set(doc)
            for word in doc:
                self.frequency[label][word] += 1
        # calculate prbability
        self.probability = []
        _max=[0]*6
        for c,f in enumerate(self.frequency):
            _max[c]= self.train_labels.count(c)
            tp = dict(zip(f.keys(), [(v + alpha) / (_max[c] + 2. * alpha) for v in f.values()]))
            self.probability.append(tp)

        # do test prediction
        test_prediction = []
        t=0
        for data in test_data:
            preds, data = [], {}.fromkeys(data)
            t += 1
            print(t)
            for i, p in enumerate(self.probability):
                # add log value instead of multiplication
                pred = math.log(self.class_probability[i])
                for word in p.keys():
                    if word in data:
                        pred += math.log(p[word])
                    else:
                        pred += math.log(1 - p[word])
                preds.append(pred)
            # do argmax to get prediction label
            preds = np.argmax(preds)
            test_prediction.append(preds)
        # calculate accuracy
        accuracy = np.sum(np.array(test_prediction) == np.array(test_labels)) / len(test_labels)
        return accuracy


data_path = "./"
# data_path = "./data/Movie Review/"
train_data, train_labels = load_data(data_path + 'train/')
test_data, test_labels = load_data(data_path + 'test/')

nb = NaiveBayes(train_data, train_labels)
# acc = nb.multinomial(test_data, test_labels)  # * Test accuracy: 0.9389920424403183
time_start=time.time()
acc = nb.bernoulli(test_data, test_labels)  # * Test accuracy: 0.8103448275862069

time_end=time.time()
print('totally cost',time_end-time_start)
print("* Test accuracy:", acc)
