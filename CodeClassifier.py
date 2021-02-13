#!/usr/bin/env python
# coding: utf-8

import re
import random
import pickle as pkl 
from itertools import groupby
from CodeClassifierTrain import rgxPattern, custom_tokenizer


random.seed(123)


class CodeClassifier:
    def __init__(self):
        with open('models/classifier.model', 'rb') as file:
            self.sgd_classifier = pkl.load(file)
        with open('models/tfIdf.model', 'rb') as file:
            self.tfidf = pkl.load(file)
        with open('models/lbEncoder.model', 'rb') as file:
            self.lbencoder = pkl.load(file)
        self.features = self.tfidf.get_feature_names()

    def classify(self, code, numfeatures):
        code = code.strip()
        prob_list = self.sgd_classifier.predict_proba(self.tfidf.transform([code]))[0]
        top3idx = prob_list.argsort()[-1:-4:-1]
        confidence = prob_list[top3idx]*100
        confidence = confidence.round().astype(int)
        langs = self.lbencoder.inverse_transform(top3idx)

        # Get top numFeatures for the most confidently predicted language
        top_features = []
        code_tokens = custom_tokenizer(code)
        for icoef in self.sgd_classifier.coef_[top3idx[0],:].argsort()[::-1]:
            if self.features[icoef] in code_tokens:
                top_features.append(self.features[icoef])
            if len(top_features)>=numfeatures:
                break
        
        return langs, confidence, top_features


if __name__ == "__main__":
    code_classifier = CodeClassifier()
    code = '\n//This is a sample getter function\nprivate int LOC = 0;\npublic int getLOC() {\n  return LOC;\n}\npublic void setLOC(int IOC) {\n  LOC = IOC;\n}\n'
    langs, confidence, top_features = code_classifier.classify(code, 10)
    print("Languages: ", langs)
    print("Confidence: ", confidence)
    print("Top Features: ", top_features)

