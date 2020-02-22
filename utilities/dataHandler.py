"""
Library for handling data to generate corpus of words - John McCambridge
"""
import os
import numpy as np
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from numpy import asarray, savetxt

SMALL_MOVIE_REVIEWS_POS = '../datasets/small_reviews/pos'
SMALL_MOVIE_REVIEWS_NEG = '../datasets/small_reviews/neg'
CORPUS_OUTPUT = '../datasets/corpus/cornell_small.txt'
VECTOR_OUTPUT = 'datasets/weights/cornell_small_vectors.csv'

def generate_corpus(positive, negative):
    corpus = set();

    locations = [positive, negative]
    stop_words = set(stopwords.words('english'))
    output = open(CORPUS_OUTPUT, "w")

    for location in locations:
        for filename in os.listdir(location):
            if filename.endswith(".txt"):
                review = word_tokenize(open(location + "/" + filename, "r", encoding="windows-1252").read())
                
                for word in review:
                    # only add if the word is alphanumeric
                    if word.isalnum() and word not in stop_words and word not in corpus:
                        corpus.add(word)

                        output.write(word + "\n")
    
    return CORPUS_OUTPUT

def get_corpus(corpus_location):
    corpus = open(corpus_location, "r").read().split("\n")
    return corpus

# caution: heavy function
def generate_vectors(words, model):
    vectors = []
    for word in words:
        # vector = [None]
        vector = np.ones(100)
        if word in model:
            vector = model[word]
        
        vectors.append(vector)
    
    data = asarray(vectors)
    savetxt(VECTOR_OUTPUT, data, delimiter=',')
    print("Successfully saved weights to file.")

def loading(size):
    print("=====================================")
    print("Words added to corpus: " + str(size))
    print("=====================================")

    os.system('cls' if os.name == 'nt' else 'clear')

        

