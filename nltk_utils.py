#Import Libraries -
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download("punkt")

#Tokenization - sentence into tokens -
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#Lemmatization - Convert word to its root form with meaning -
def lemm(word):
    return lemmatizer.lemmatize(word.lower())

#Converting words to its vector form -
"""return bag of words array: for each known word that exists in the sentence, 0 otherwise
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]"""

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemm(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
 