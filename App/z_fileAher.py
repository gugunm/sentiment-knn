#!/home/mediamer/ta/bin python
# coding: utf-8

# # Import semua library dan data



import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

file = open("")
df = pd.read_csv(file)
df = pd.DataFrame(data = df)
df.head()



df = df[['objek','ulasan','label']]
df.head()



df['ulasan'] = df['ulasan'].str.lower()
df.head()



import re
df['ulasan'] = df['ulasan'].apply(lambda x : re.sub(r"(?:{})|([^\w\s.]+|_+)",'',x))#remove selain titik seperti underscore
df['ulasan'] = df['ulasan'].apply(lambda x : re.sub(r'[^\w\s]',' ',x))
df.head()





factory = StemmerFactory()
stemmer = factory.create_stemmer()


df['stemming'] = df['ulasan'].apply(lambda x : stemmer.stem(x))
df.head()





df['token'] = df['ulasan'].apply(lambda x : x.split())
df.head()





df['token'][1]






import re, string
regex = re.compile('[%s]'% re.escape(string.punctuation))
def remove_punctuation(s):
    return regex.sub('', s)




df["new_column"] = df['ulasan'].str.replace('[^\w\s]','')




df.head()




import pandas as pd
dk = pd.read_csv(open('AIR_PANAS_BANJAR.csv'))
dk.head()



dk['ulasan'][1]





dk['ulasan'][1].lower()



# first string
firstString = "abc"
secondString = "ghi"
thirdString = "ab"

string = "abcdef"
print("Original string:", string)

translation = string.maketrans(firstString, secondString, thirdString)

# translate string
print("Translated string:", string.translate(translation))




s = 'Datanglah ke airpanas ^_^  #@banjar..sebelum mandi coba dulu segelas kopi bali dengan pisang gorengnya atau fried banana...berbagai menu ada..makanan dan minuman..lets go'

import re
s = re.sub(r"(?:{})|([^\w\s.]+|_+)",'',s)#remove selain titik seperti underscore
s = re.sub(r'[^\w\s]',' ',s)#remove titik
print(s)



import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('dataKata.txt').read()))

def P(word, N=sum(WORDS.values())): 
    # "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    # "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    # "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    # "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    # "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)] # [('', 'kemarin'), ('k', 'emarin'), ('ke', 'marin'), dst]
    deletes    = [L + R[1:]               for L, R in splits if R] # ['emarin', 'kmarin', 'kearin', dst]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1] # ['ekmarin', 'kmearin', 'keamrin', dst]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters] # ['aemarin', 'bemarin', 'cemarin', dst]
    inserts    = [L + c + R               for L, R in splits for c in letters] # ['akemarin', 'bkemarin', 'ckemarin', dst]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    # "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

kata = 'syg'
print('kata typo : ', kata)
print('koreksi : ', correction(kata))



from collections import Counter


WORDS = Counter(words(open('dataKata.txt').read()))

def P(word, N=sum(WORDS.values())): 
    # "Probability of `word`."
    return WORDS[word] / N

print(P('mkan'))
print(WORDS['mkan'])
