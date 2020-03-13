"""
Algorithm for classifying words based on SentiGraph.
SentiNode
SentiGraph: 
""" 
import time
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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

from progress.bar import Bar

EMOLEX_PATH = "datasets/emolex/emolex.txt"

print("Loading Sentiment Graph...")
G = nx.read_gpickle("graph_outputs/iterations/v3/small_epsilon_small_endpoint_radius_and_cost_pickled_output_14100_wo_neutral.gpickle")

# PROFESSOR_REVIEWS = "datasets/corpus/professor_reviews_full.txt"

# reviews = open(PROFESSOR_REVIEWS).read().split("@")
# information = {}

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

"""
Converts a paragraph of text into bag format of tokenized sentences
"""
def passage_to_bag(passage):
    bag = []

    # stop words will reduce words being classified including 'the', 'and' etc
    stop_words = set(stopwords.words('english'))

    # split the passage into a series of tokenized sentences
    sentences = sent_tokenize(passage)

    bar = Bar('Parsing Sentence', max=len(sentences))

    for sentence in sentences:
        # split each sentence into a series of tokenized words
        tokenized = word_tokenize(sentence)

        subBag = []

        for word in tokenized:
            if word not in stop_words:
                subBag.append(word.lower())
        
        bag.append(subBag)
        
        bar.next()
        time.sleep(0.01)

    bar.finish()

    return bag

"""
Given a score (as a hash map) returns the smallest value
"""

def largest(score):
    largest = float("-inf")
    label = None

    for item in score:
        if score[item] > largest:
            largest = score[item]
            label = item

    return label

"""
Given a list of scores from each sentence, will generate the necessary classification.
"""

def score_phrase(scores):
    rating = {}
    # bar = Bar('Scoring Passage', max=len(scores))

    normalization_scores = {}

    for score, gamma in scores:

        # extract the min (smallest path from word to emotion) classification
        classificiation = min(score)

        calculated_score = classificiation[0]
        label = classificiation[1]

        if label in normalization_scores:
            normalization_scores[label] += 1
        else:
            normalization_scores[label] = 1

        if label in rating:
            rating[label] += (calculated_score * gamma)
        else:
            rating[label] = (calculated_score * gamma)
    
    if len(scores) > 0:
        for key in rating:
            rating[key] = float(rating[key]) / float(normalization_scores[key])

    return rating

def classify(passage):
    # convert the passage into sentences of keywords

    sentence_keywords = passage_to_bag(passage)    
    
    # pronouns to detect if the sentence is relative to the professor
    # if so, increase the gamma
    pronouns = ["he", "him", "she", "her", "prof", "professor"]
    
    gamma = 1.0
    isProfessorSpecific = False

    score = {
        "#-ANGER": 0,
        "#-JOY": 0,
        "#-SAD": 0, 
    }
    
    i = 1
    print("====================================")
    for sentence in sentence_keywords:
        scores = []
          
        # generate a score for each sentence
        for keyword in sentence:
            if keyword in pronouns:
                isProfessorSpecific = True
            
            if isProfessorSpecific:
                gamma = 1.5

            if G.has_node(keyword.lower()):
                types = ["#-ANGER", "#-JOY", "#-SAD"]
                paths = []

                for emotion in types:
                    try:
                        path = (nx.dijkstra_path_length(G, source=emotion, target=keyword.lower()), emotion)
                        paths.append(path)
                    except:
                        print("No path between " + emotion + " and " + str(keyword))

                if len(paths) > 0:
                    scores.append((paths, gamma))
                else:
                    print("# Cannot classify: " + keyword)

        gamma = 1
        isProfessorSpecific = False

        emotional_scorings = score_phrase(scores)

        for emotion_key in emotional_scorings:
            score[emotion_key] += emotional_scorings[emotion_key]
    
        # print(" * Sentence '" + str(" ".join(sentence)) + "' Output = " + str(emotional_scorings))
        i += 1

    for item in score:
        if score[item] != 0:
            score[item] = 1.0 / float(score[item])

    classification_label = largest(score)

    print(" *** Passage has sentiment: " + str(score) + " - " + str(classification_label))
    
    return (score, classification_label)