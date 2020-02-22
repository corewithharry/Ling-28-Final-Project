"""
Library for handling data to generate corpus of words - John McCambridge
"""
import os
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords

SMALL_MOVIE_REVIEWS_POS = '../datasets/small_reviews/pos'
SMALL_MOVIE_REVIEWS_NEG = '../datasets/small_reviews/neg'
CORPUS_OUTPUT = '../datasets/corpus/cornell_small.txt'

def generate_corpus(positive, negative):
    corpus = set();

    locations = [positive, negative]
    stop_words = set(stopwords.words('english'))

    for location in locations:
        for filename in os.listdir(location):
            if filename.endswith(".txt"):
                review = word_tokenize(open(location + "/" + filename, "r", encoding="windows-1252").read())
                
                for word in review:
                    # only add if the word is alphanumeric
                    if word.isalnum() and word not in stop_words and word not in corpus:
                        corpus.add(word)
                        # loading(len(corpus))
                        


def loading(size):
    print("=====================================")
    print("Words added to corpus: " + str(size))
    print("=====================================")

    os.system('cls' if os.name == 'nt' else 'clear')

corpus = generate_corpus(SMALL_MOVIE_REVIEWS_POS, SMALL_MOVIE_REVIEWS_NEG)
print(corpus)

        

