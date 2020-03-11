"""
Algorithm for classifying words based on SentiGraph.
SentiNode
SentiGraph: 
""" 
import time
import random
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

print("Loading in Graph...")
G = nx.read_gpickle("graph_outputs/iterations/v2/large_epsilon_small_endpoint_radius_and_cost_pickled_output_14100_wo_neutral.gpickle")

PROFESSOR_REVIEWS = "datasets/corpus/professor_reviews_full.txt"

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

# quite a joyful example with subtle negatives
joyExample = """This class is very interesting and you will learn a lot!! However, all of the irrelevant material and random statistics are tested on the exams. The grades at the end of the course are decided arbitrarily, and Sargent said he did not feel comfortable giving me an A- in the course when my final percentage was a 90.5%, so he gave me a B. I definitely recommend this class, but would highly recommend setting it as an NRO. Sargent is a really cool guy, and does a good job of incorporating public health, medicine, and adolescent health. When you give your group final presentation in the last week of the course, Sargent likes it if you follow the same format that he gave for his lectures. Don't do the readings for the class...Although he will tell you otherwise, Dr. Sargent covers all of the important readings in detail during class.", "This class is very interesting but should NOT be considered a layup. You can get away with doing very little work (outside of studying for exams), if you're ok with getting a B. If you want to get an A, you need to spend a lot of time on every reading, pay attention & take diligent notes in every class, and it is definitely time-consuming. There's a lot of reading studies, and the exams demand that you remember some of the more minute details of those studies. Study hard for the exams, they are not easy. There is also a lot of group project work, which isn't super difficult but involves presenting in front of the class frequently. Professor Sargent is a really nice guy and, in my opinion, a great prof so overall I would recommend the course."""

angerExample = """AVOID AT ALL COSTS! Does this man know physics, it seems so. Can he teach it? No! The lectures are extremely unorganized, and he often can't figure out how to solve his own example problems. He'll go on tangents about topics not included in the exams and then never teach the things that are being tested, so you have to rely on TA office hours or the textbook to figure out the content. Someone once asked "what is weight" to which he responded "define it however you want." By the end of the term most students did not show up to class since the lectures were so useless that they made you more confused. I can't say whether the class itself is good or bad, but my warning is to not take it with Whitfield."""

sadExample = """Sitting professor X's class always filled me with strong depression and sadness; I would cry every week during the X-hours and the p-sets were so difficult it added to my constant class depression; it is quite tragic and emotional how bad this class made me feel."""

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
Given a list of scores from each sentence, will generate the necessary classification.
"""

def score_phrase(scores):
    rating = {}
    bar = Bar('Scoring Passage', max=len(scores))

    for score, gamma in scores:

        # extract the min (smallest path from word to emotion) classification
        classificiation = min(score)

        calculated_score = classificiation[0]
        label = classificiation[1]

        if label in rating:
            rating[label] += (calculated_score * gamma)
        else:
            rating[label] = (calculated_score * gamma)

        bar.next()
        time.sleep(0.01)

    bar.finish()
    
    if len(scores) > 0:
        for key in rating:
            rating[key] = rating[key] / len(scores)

        # the ratings are the summation of path lengths from a word to emotion
        # the inverse is taken as small values should have big scores
        for key in rating:
            rating[key] = 1 / rating[key]

    return rating

def classify(passage):
    # convert the passage into sentences of keywords

    sentence_keywords = passage_to_bag(passage)    
    
    # pronouns to detect if the sentence is relative to the professor
    # if so, increase the gamma
    pronouns = ["he", "him", "she", "her", "prof", "professor"]
    
    gamma = 1.0
    isProfessorSpecific = False

    i = 0
    print("====================================")
    for sentence in sentence_keywords:
        scores = []
        
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
                        path = (nx.dijkstra_path_length(G, source=emotion, target=keyword), emotion)
                        paths.append(path)
                    except:
                        print("No path between " + emotion + " and " + str(keyword))

                if len(paths) > 0:
                    # print("# Keyword: '" + keyword + "' has emotion: " + str(paths))
                    scores.append((paths, gamma))
                else:
                    print("# Cannot classify: " + keyword)

        gamma = 1
        isProfessorSpecific = False
        print(" * Sentence '" + str(" ".join(sentence)) + "' Output = " + str(score_phrase(scores)))
        i += 1

    # todo:
    # check if word exists within the graph - DONE
    # find the closest end-point - DONE
    # multiply by gamma 

passages = [("# Sad Example", sadExample)]

for passage in passages:
    print("Classifying: " + passage[0])
    classify(passage[1])