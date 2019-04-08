from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import numpy as np
import glob
import math
import os
import re
import warnings

warnings.filterwarnings("ignore")

def preprocessing(data_path=None): #, stopword=stopword, stemmer=stemmer):
    tokenizer = RegexpTokenizer(r'\w+')

    arr_praproses = list()

    datafiles = glob.glob(data_path)

    for i, name in enumerate(datafiles):      #Looping file .xml yang ada di file /data/train
        df = pd.read_csv(name, encoding='"ISO-8859-1"')
        df = df.dropna()
        if i == 0:
            da = df.copy()
        else:
            da.append(df)

    lUlasan = da["Ulasan"].values.tolist()
    lLabel = da["Label"].values.tolist()
    # replace positif = 1 and negatif = 0
    lLabel = [1 if x == 'Postif' or x == 'Positif' else 0 for x in lLabel]

    for ulasan in lUlasan:
        lowcase_word = ulasan.lower()       #case folding lowcase data perbaris
        tokens = tokenizer.tokenize(lowcase_word)       #Tokenisasi Kalimat, tergantung proses terakhirnya, stemming atau stopword atau hanya casefolding
        filtered_words = [w for w in tokens if not w in stopwords.words('english')]     #remove Stopwords
        output = list()       
        for kata in filtered_words:
            # output.append(PorterStemmer().stem(kata)) #proses stemming per-kata dalam 1 kalimat
            output.append(WordNetLemmatizer().lemmatize(kata)) #proses lemmatisasi per-kata dalam 1 kalimat
        # sentence = " ".join(output) + ''
        arr_praproses.append(output)                #tampung kalimat hasil stemm ke arr_praproses
    
    df = pd.DataFrame({"ulasan" : arr_praproses, "label" : lLabel})
    return df

# data_path = 'data/input/*.csv'
# df = preprocessing(data_path=data_path)
# print(df)