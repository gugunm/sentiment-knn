from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import glob
import warnings

warnings.filterwarnings("ignore")

class Preprocessing:
    def __init__(self):
        # initialize tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

    def cleandata(self, path=None, stemmer=False, lemmatizer=False):
        # initialize list of files
        files = glob.glob(path)
        arr_praproses = []
        # combine files to be one dataframe
        for i, name in enumerate(files):
            df = pd.read_csv(name, encoding='"ISO-8859-1"')
            df = df.dropna()
            if i == 0:
                da = df.copy()
            else:
                da.append(df)
        # transform column ulasan tolist
        lUlasan = da["Ulasan"].values.tolist()
        # transform column label tolist
        lLabel = da["Label"].values.tolist()
        # replace positif = 1 and negatif = 0
        lLabel = [1 if x == 'Postif' or x == 'Positif' else 0 for x in lLabel]

        for ulasan in lUlasan:
            # case folding (lowcase) sentence
            lowcase_word = ulasan.lower()
            # tokenize sentence
            tokens = self.tokenizer.tokenize(lowcase_word)
            # remove stopwords of english
            filtered_words = [w for w in tokens if not w in stopwords.words('english')]
            # initialize list of output
            output = []       
            for kata in filtered_words:
                # using steming, if stemmer True
                if stemmer :
                    output.append(PorterStemmer().stem(kata))
                # using lemmatizer, if lemmatizer true
                elif lemmatizer :
                    output.append(WordNetLemmatizer().lemmatize(kata))
                # if not using both
                else :
                    output.append(kata)
            # append sentence to array of sentences
            arr_praproses.append(output)
        # initialize dict of clean data 
        data = {
            "ulasan" : np.array(arr_praproses),
            "label" : np.array(lLabel)
        }

        return data

    def cleanambiguity(data=None):
        pass
