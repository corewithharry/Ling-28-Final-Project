"""
Algorithm for generating bidirected graph relating words with sentiment and context.
SentiNode
SentiGraph: 
""" 
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
from numpy import loadtxt
from gensim.models import Word2Vec
from nltk.corpus import brown
from utilities.vector import distance, similarity
from utilities.dataHandler import get_corpus, generate_vectors, generate_MLE, generate_emolex_vectors
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.lm import MLE, NgramCounter
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

CORPUS_LOCATION = "datasets/corpus/cornell_small.txt"
MODEL_PATH = "models/BrownWord2Vec.model"
VECTOR_PATH = "datasets/weights/cornell_small_vectors.csv"
EMOLEX_PATH = "datasets/emolex/emolex.txt"

G = nx.Graph()

# maximum Euclidean distance between two words
epsilon = 2.0
cost_threshold = 0.95 # 0.6 # the maximum edge weight (1 - cos similarity) i.e. closer to 0 means similar, 1 means totally not similare
endpoint_radius = 0.25 # 0.85 # minimum cosine similarity

emolex = generate_emolex_vectors(EMOLEX_PATH)

max_vec = float("-inf")
max_word = None

locations = emolex.keys()

# maximum number of words inside a graph
maximum = 14100
i = 0

# total summation of a neutral vector
neutral = 0.09999999999999999

start = time.time()

# vector for anger, joy, and sadness
G.add_node("#-ANGER")
G.add_node("#-JOY")
G.add_node("#-SAD")

# anger is a partial linear combination of sad fix this and retrain

# anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust

endpoints = {
    "#-ANGER": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "#-JOY":   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "#-SAD":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}

k = 0

for w_i in emolex:
    w_i = word_tokenize(w_i)[0]
    if sum(emolex[w_i]) != neutral:
        G.add_node(w_i)

        # similarity scores between the word and each endpoint
        anger_score = similarity(emolex[w_i], endpoints["#-ANGER"])
        joy_score = similarity(emolex[w_i], endpoints["#-JOY"])
        sad_score = similarity(emolex[w_i], endpoints["#-SAD"])

        # determine if the current word is in the radius of an endpoint
        if anger_score > endpoint_radius:
            G.add_edge(w_i, "#-ANGER", weight=anger_score)

        if joy_score > endpoint_radius:
            G.add_edge(w_i, "#-JOY", weight=joy_score)

        if sad_score > endpoint_radius:
            G.add_edge(w_i, "#-SAD", weight=sad_score)

        # connect to every other node in the graph
        for w_j in list(G.nodes):
            if w_i != w_j and w_j not in endpoints:
                # connect w_i and w_j together with edge weight of similarity

                if distance(emolex[w_i], emolex[w_j]) < epsilon:
                    similarity_score = similarity(emolex[w_i], emolex[w_j])
                    similarity_cost = max(1 - similarity_score, 0.01)

                    if similarity_cost < cost_threshold:
                        G.add_edge(w_i, w_j, weight=similarity_cost)
                        
                    # G.add_edge(w_j, w_i, weight=1/similarity_cost)
        print("Adding Node: '" + w_i + "' - No. " + str(k) + " w. " + str(G.number_of_edges()) + " edges / " + str(time.time() - start) + " sec(s) elapsed.")
        k += 1            
        

    if i == maximum:
        break

    i += 1

# nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
# nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
# plt.show()

# started: 4:02pm
# after 1 minute: 2500 nodes inserted
# 4:05pm: 4100 nodes inserted

# nx.write_graphml(G, "graph_outputs/tokenized_output_" + str(maximum) + "_wo_neutral.graphml")
nx.write_gpickle(G, "graph_outputs/iterations/v3/small_epsilon_small_endpoint_radius_and_cost_pickled_output_" + str(maximum) + "_wo_neutral.gpickle")