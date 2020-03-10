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

bar = Bar('Loading Graph', max=100)
G = nx.read_gpickle("graph_outputs/pickled_output_14100_wo_neutral.gpickle")

for i in range(0, 100):
    bar.next()

bar.finish()

time.sleep(1)

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

# quite a joyful example with subtle negatives
joyExample = """This class is very interesting and you will learn a lot!! However, all of the irrelevant material and random statistics are tested on the exams. The grades at the end of the course are decided arbitrarily, and Sargent said he did not feel comfortable giving me an A- in the course when my final percentage was a 90.5%, so he gave me a B. I definitely recommend this class, but would highly recommend setting it as an NRO. Sargent is a really cool guy, and does a good job of incorporating public health, medicine, and adolescent health. When you give your group final presentation in the last week of the course, Sargent likes it if you follow the same format that he gave for his lectures. Don't do the readings for the class...Although he will tell you otherwise, Dr. Sargent covers all of the important readings in detail during class.", "This class is very interesting but should NOT be considered a layup. You can get away with doing very little work (outside of studying for exams), if you're ok with getting a B. If you want to get an A, you need to spend a lot of time on every reading, pay attention & take diligent notes in every class, and it is definitely time-consuming. There's a lot of reading studies, and the exams demand that you remember some of the more minute details of those studies. Study hard for the exams, they are not easy. There is also a lot of group project work, which isn't super difficult but involves presenting in front of the class frequently. Professor Sargent is a really nice guy and, in my opinion, a great prof so overall I would recommend the course."""

angerExample = """AVOID AT ALL COSTS! Does this man know physics, it seems so. Can he teach it? No! The lectures are extremely unorganized, and he often can't figure out how to solve his own example problems. He'll go on tangents about topics not included in the exams and then never teach the things that are being tested, so you have to rely on TA office hours or the textbook to figure out the content. Someone once asked "what is weight" to which he responded "define it however you want." By the end of the term most students did not show up to class since the lectures were so useless that they made you more confused. I can't say whether the class itself is good or bad, but my warning is to not take it with Whitfield."""

sadExample = """Sitting professor X's class always filled me with strong depression and sadness; I would cry every week during the X-hours and the p-sets were so difficult it added to my constant class depression; it is quite tragic and emotional how bad this class made me feel."""

def passage_to_bag(passage):
    bag = []
    stop_words = set(stopwords.words('english'))
    tokenized = word_tokenize(passage)

    bar = Bar('Parsing Sentence', max=len(tokenized))

    # convert to bag of words
    for word in tokenized:
        if word not in stop_words:
            if G.has_node(word.lower()):
                bag.append(word.lower())
        bar.next()
        time.sleep(0.01)
    bar.finish()

    return bag

# takes approx 30 seconds to classify one word ... fuck

"""
# Keyword: interesting has emotion: (1.285520784778797, '#-JOY')
# Keyword: learn has emotion: (1.285520784778797, '#-JOY')
# Keyword: irrelevant has emotion: (1.285520784778797, '#-ANGER')
# Keyword: giving has emotion: (1.285520784778797, '#-JOY')
# Keyword: recommend has emotion: (1.3585900955005383, '#-JOY')
# Keyword: recommend has emotion: (1.3585900955005383, '#-JOY')
# Keyword: cool has emotion: (1.285520784778797, '#-JOY')
# Keyword: good has emotion: (1.357860058863077, '#-JOY')
# Keyword: job has emotion: (1.285520784778797, '#-JOY')
# Keyword: public has emotion: (1.3585900955005386, '#-JOY')
# Keyword: important has emotion: (1.3585900955005383, '#-JOY')
# Keyword: interesting has emotion: (1.285520784778797, '#-JOY')
# Keyword: time has emotion: (1.5902273736421795, '#-ANGER')
# Keyword: reading has emotion: (1.285520784778797, '#-JOY')
# Keyword: pay has emotion: (1.285606476604563, '#-JOY')
# Keyword: attention has emotion: (1.285520784778797, '#-JOY')
# Keyword: reading has emotion: (1.285520784778797, '#-JOY')
# Keyword: demand has emotion: (0.9998000599800069, '#-ANGER')
# Keyword: study has emotion: (1.285520784778797, '#-JOY')
# Keyword: difficult has emotion: (1.5902273736421793, '#-ANGER')
# Keyword: professor has emotion: (1.3585900955005383, '#-JOY')
# Keyword: recommend has emotion: (1.3585900955005383, '#-JOY')
"""

def classify(passage):
    keywords = passage_to_bag(passage)    

    print("====================================")
    for keyword in keywords:
        types = ["#-ANGER", "#-JOY", "#-SAD"]
        paths = []

        for emotion in types:
            try:
                path = (nx.dijkstra_path_length(G, source=emotion, target=keyword), emotion)
                paths.append(path)
            except:
                print("No path between " + emotion + " and " + str(keyword))

        if len(paths) > 0:        
            print("# Keyword: " + keyword + " has emotion: " + str(min(paths)))
        else:
            print("# Cannot classify: " + keyword)

    return keywords
    # todo:
    # check if word exists within the graph
    # find the closest end-point
    # multiply by gamma 

passages = [("# Joy Example", joyExample), ("# Angry Example", angerExample), ("# Sad Example", sadExample)]

for passage in passages:
    print("Classifying: " + passage[0])
    classify(passage[1])