import re
import string
from nltk import word_tokenize as lib_tokenizer

import nltk
nltk.download('punkt')

dict_map = dict({})  
def word_tokenize(text):
    words = text.split() 
    words_norm = [] 
    for w in words: 
        if dict_map.get(w, None) is None: 
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"') 
        words_norm.append(dict_map[w]) 
    return words_norm 

def strip_context(text):
    text = text.replace('\n', ' ') 
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip() 
    return text

def post_process(x):
    x = " ".join(word_tokenize(strip_context(x))).strip()
    x = x.replace("\n"," ")
    x = "".join([i for i in x if i not in string.punctuation])
    return x

def preprocess(x, max_length=-1, remove_puncts=False):
    x = nltk_tokenize(x)
    x = x.replace("\n", " ")
    if remove_puncts:
        x = "".join([i for i in x if i not in string.punctuation])
    if max_length > 0:
        x = " ".join(x.split()[:max_length])
    return x

def nltk_tokenize(x):
    return " ".join(word_tokenize(strip_context(x))).strip()
