import pandas as pd
import numpy as np
import math

class FeatureExtraction:
    def __init__(self):
        # fitur untuk seluruh dokumen ()
        self.allDocsFeatures = dict()
        # array, isinya sebanyak data yang digunakan. dan setiap isinya mengandung fitur tiap dokumen (TF)
        self.oneDocFeature   = list()

    # create dict to save word frequency for each document
    def one_doc_feature(self, sentence):
        feature = dict()
        for word in sentence:
            if(word not in feature):
                feature[word] = 1
            else:
                feature[word] += 1
        return feature

    # create dict to save word frequency for all document
    def all_doc_feature(self, feature, allDocsFeatures):
        for i, f in enumerate(feature):
            if f not in allDocsFeatures:
                allDocsFeatures[f] =  1
            else:
                allDocsFeatures[f] += 1
        return allDocsFeatures

    # get features > for each doc, and all docs
    def get_features(self, data, oneDocFeature, allDocsFeatures):        
        for row in data:
            # Tipe dict isinya fitur dan frekuensi per dokumen | katanya udah unik
            feature = self.one_doc_feature(row)    
            # dict dimasukin ke array bernama fitur_onedoc
            oneDocFeature.append(feature)
            # fitur perdoc dimasukin ke fitur alldoc dalam bentuk dict      
            allDocsFeatures = self.all_doc_feature(feature, allDocsFeatures)
    
    # create term frequency
    def tf(self, bow, oneDocFeature):
        # initialize tf table
        tf_table = np.zeros((len(oneDocFeature), len(bow)), dtype=int)
        for n_doc, doc in enumerate(oneDocFeature):
            for n_feature, f in enumerate(bow):
                if(f in doc):
                    tf_table[n_doc,n_feature] = doc[f]
        return tf_table

    # create idf dict
    def idf(self, oneDocFeature, size_docs):
        for feature in oneDocFeature:
            oneDocFeature[feature] = math.log10(size_docs/oneDocFeature[feature])

    # create tf-idf table
    def tf_idf(self, tf, idf, bow):
        tfidf = np.zeros((len(tf), len(idf)), dtype=float)
        for i in range(len(tf)):
            for j, fitur in enumerate(bow):
                tfidf[i,j] = tf[i,j]*idf[fitur]
        return tfidf

    # main program of tf-idf process
    def get_tf_idf(self, data):
        # get feature for one doc and all docs
        self.get_features(data, self.oneDocFeature, self.allDocsFeatures)
        # save dict keys to list (bag of words)
        bow = list(self.allDocsFeatures.keys())
        # inverse document frequency table
        self.idf(self.allDocsFeatures, len(self.oneDocFeature))
        # term frequency table
        tf_table  = self.tf(bow, self.oneDocFeature)
        # tf-idf table
        tfidf = self.tf_idf(tf_table, self.allDocsFeatures, bow)
        return tfidf