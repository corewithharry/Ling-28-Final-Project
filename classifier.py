"""
Algorithm for classifying words based on SentiGraph.
SentiNode
SentiGraph: 
""" 
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
from numpy import loadtxt
from gensim.models import Word2Vec
from utilities.vector import distance, similarity
from utilities.dataHandler import get_corpus, generate_vectors, generate_MLE, generate_emolex_vectors

from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.lm import MLE, NgramCounter
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

G = nx.read_graphml("graph_outputs/output_1000_wo_neutral.graphml", node_type=str)

PROFESSOR_REVIEWS = "datasets/corpus/professor_reviews_full.txt"

reviews = open(PROFESSOR_REVIEWS).read().split("@")
information = {}

# categorize the data into PROFESSOR -> { COURSE_ID -> [ REVIEWS ]}
"""
for review in reviews:
    review = review.split("\n")
    if len(review) == 5:
        professor = review[1]
        course_id = review[2]
        data = review[3]
        
        if professor not in information:
            information[professor] = {}
        
        if course_id not in information[professor]:
            information[professor][course_id] = [data]
        else:
            information[professor][course_id].append(data)
"""

example = """This class is very interesting and you will learn a lot!! However, all of the irrelevant material and random statistics are tested on the exams. The grades at the end of the course are decided arbitrarily, and Sargent said he did not feel comfortable giving me an A- in the course when my final percentage was a 90.5%, so he gave me a B. I definitely recommend this class, but would highly recommend setting it as an NRO. Sargent is a really cool guy, and does a good job of incorporating public health, medicine, and adolescent health. When you give your group final presentation in the last week of the course, Sargent likes it if you follow the same format that he gave for his lectures. Don't do the readings for the class...Although he will tell you otherwise, Dr. Sargent covers all of the important readings in detail during class.", "This class is very interesting but should NOT be considered a layup. You can get away with doing very little work (outside of studying for exams), if you're ok with getting a B. If you want to get an A, you need to spend a lot of time on every reading, pay attention & take diligent notes in every class, and it is definitely time-consuming. There's a lot of reading studies, and the exams demand that you remember some of the more minute details of those studies. Study hard for the exams, they are not easy. There is also a lot of group project work, which isn't super difficult but involves presenting in front of the class frequently. Professor Sargent is a really nice guy and, in my opinion, a great prof so overall I would recommend the course."""

def classify(passage):
    bag = []
    stop_words = set(stopwords.words('english'))

    for word in word_tokenize(passage):
        if word not in stop_words:
            bag.append(word.lower())

    return bag

print(classify(example))