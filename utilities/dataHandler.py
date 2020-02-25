"""
Library for handling data to generate corpus of words - John McCambridge
"""
import os
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.lm import MLE, NgramCounter
from nltk.util import ngrams
from nltk.corpus import stopwords
from numpy import asarray, savetxt
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

SMALL_MOVIE_REVIEWS_POS = '../datasets/small_reviews/pos'
SMALL_MOVIE_REVIEWS_NEG = '../datasets/small_reviews/neg'
CORPUS_OUTPUT = '../datasets/corpus/cornell_small.txt'
VECTOR_OUTPUT = 'datasets/weights/cornell_small_vectors.csv'\

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

def generate_MLE(positive, negative):
    tokenized = []
    text = ""
    n = 2 # n-gram size

    locations = [positive, negative]
    for location in locations:
        for filename in os.listdir(location):
            if filename.endswith(".txt"):
                contents = open(location + "/" + filename, "r", encoding="windows-1252").read()
                review = sent_tokenize(contents)
                for word in review:
                    tokenized.append(word.lower())
                    
                text += contents + " "

    paddedLine = [list(pad_both_ends(word_tokenize(text.lower()), n))]
    train, vocab = padded_everygram_pipeline(n, paddedLine)

    model = MLE(n)
    model.fit(train, vocab)

    return model

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

# generates a hash-map of emotional 
def generate_emolex_vectors(emolex_location):
    NRCEmotions = open(emolex_location)
    NRCEmotionsData = NRCEmotions.read().split("\n")

    words = {}

    for line in NRCEmotionsData:
        packet = line.split("\t")
        if len(packet) == 3:    
            word = packet[0]
            association = packet[1]
            rating = int(packet[2])

            if rating == 0:
                rating = 0.01
            

            if word not in words:
                words[word] = [rating]
            else:
                words[word].append(rating)

    return words

def loading(size):
    print("=====================================")
    print("Words added to corpus: " + str(size))
    print("=====================================")

    os.system('cls' if os.name == 'nt' else 'clear')

        

