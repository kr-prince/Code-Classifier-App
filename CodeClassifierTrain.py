#!/usr/bin/env python
# coding: utf-8

import re
import time
import random
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt 
from itertools import groupby
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

random.seed(123)


# Regex for tokenization
rgxPattern = re.compile(r'\w+|[^\w\s]')

def custom_tokenizer(text):
    tokenList = rgxPattern.findall(text)
    # same tokens occurring recurrently are grouped together. Ex ++, ==
    tokenList = [''.join(list(group)) for _,group in groupby(tokenList)]
    return tokenList


def confusion_matrix_image(y_true, y_pred, labels, savePath=None):
    plt.figure(figsize=(16.0, 14.0))
    conf_matrix = np.round(confusion_matrix(y_true, y_pred, normalize='true'), 1)
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    hMap = sns.heatmap(df_cm, annot=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    if savePath is not None:
        plt.savefig(savePath)
    plt.close()


def train_codeClassifier(filePath):
    print("----------------- Training started -----------------")
    start_time = time.time()
    # Read the data file 
    # data is pickled form of dict where keys are languages and values are list
    # of code snippets
    with open(filePath, 'rb') as file:
        data = pkl.load(file)
    print("Data reading completed..")
    
    # Languages covered
    print("Languages:")
    print(', '.join([lang.title() for lang in data.keys()]))

    # preparing training and test data in a balanced way grouped on all the languages 
    tfIdf_traindata = []
    train_data, test_data = [], []
    for lang in data.keys():
        n = int(len(data[lang])*0.2)
        train_data.extend([(code,lang) for code in data[lang][:-n]])
        test_data.extend([(code,lang) for code in data[lang][-n:]])
        # using a small hack here to keep all the code mapped to any single 
        # language as a single document
        tfIdf_traindata.append('\n'.join(data[lang][:-n]))
    print("Training and Test data separated..")

    # shuffle data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # separating code part and target language labels
    codeList_train, langList_train=map(list, zip(*train_data))
    codeList_test, langList_test = map(list, zip(*test_data))

    # fitting language label encoder and Tf-Idf vectors
    lbencoder = LabelEncoder()
    lbencoder.fit(langList_train)
    
    tfidf = TfidfVectorizer( encoding='latin-1', sublinear_tf=True, token_pattern=None,
                    ngram_range=(1, 2), max_features=120000, tokenizer=custom_tokenizer)
    tfidf.fit(tfIdf_traindata)
    print("Language labels and vectors model fitting done..")
    
    # converting code and labels to vectors
    X_train = tfidf.transform(codeList_train)
    Y_train = lbencoder.transform(langList_train)
    X_test = tfidf.transform(codeList_test)
    Y_test = lbencoder.transform(langList_test)
    print("Train and Test data converted to vectors..")
    
    # defining and fitting classifier
    sgd_classifier = SGDClassifier( loss='modified_huber', alpha=0.00001, early_stopping=True, 
                                   penalty='elasticnet', max_iter=1500, random_state=123, 
                                   validation_fraction=0.25, epsilon=0.01)

    sgd_classifier.fit(X_train, Y_train)
    print("Classifier fitting done on training data..")
    
    Y_pred_train = sgd_classifier.predict(X_train)
    train_acc = round(accuracy_score(Y_train, Y_pred_train)*100, 2)
    print("Training Accuracy : {0}%".format(train_acc))
    
    Y_pred=sgd_classifier.predict(X_test)
    test_acc = round(accuracy_score(Y_test, Y_pred)*100, 2)
    print("Test Accuracy : {0}%".format(test_acc))
    
    with open('results/classification_report_train.txt','w') as cr:
        cr.write(classification_report(Y_train, Y_pred_train, target_names=lbencoder.classes_))
    print("Training Classification report saved..")        
    
    confusion_matrix_image(Y_train, Y_pred_train, lbencoder.classes_, 'results/confusion_matrix_train.png')
    print("Training Confusion matrix saved..")
    
    with open('results/classification_report_test.txt','w') as cr:
        cr.write(classification_report(Y_test, Y_pred, target_names=lbencoder.classes_))
    print("Test Classification report saved..")        
    
    confusion_matrix_image(Y_test, Y_pred, lbencoder.classes_, 'results/confusion_matrix_test.png')
    print("Test Confusion matrix saved..")
    
    with open('models/classifier.model', 'wb') as file:
        pkl.dump(sgd_classifier, file)
    with open('models/tfIdf.model', 'wb') as file:
        pkl.dump(tfidf, file)
    with open('models/lbEncoder.model', 'wb') as file:
        pkl.dump(lbencoder, file)
    print("Classifier, Tf-Idf and LabelEncoder models saved..")
    
    tmins = round(((time.time() - start_time)/60),1)
    print("---------- Training Completed in {0} mins ----------".format(tmins))


if __name__ == "__main__":
    filePath='./data/snippets.cpkl'
    train_codeClassifier(filePath)




