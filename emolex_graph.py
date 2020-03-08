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

from nltk.lm import MLE, NgramCounter
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

CORPUS_LOCATION = "datasets/corpus/cornell_small.txt"
MODEL_PATH = "models/BrownWord2Vec.model"
VECTOR_PATH = "datasets/weights/cornell_small_vectors.csv"
EMOLEX_PATH = "datasets/emolex/emolex.txt"


# brown_corpus = brown.sents()
model = Word2Vec.load(MODEL_PATH) # Word2Vec(brown_corpus)
# returns a list of uncategorized words in the corpus
corpus = get_corpus(CORPUS_LOCATION)
vectors = loadtxt(VECTOR_PATH, delimiter=',')

G = nx.Graph()

# word to vec hash map
w2v = dict()

# maximum Euclidean distance between two words
epsilon = 2.0
cost_threshold = 0.6 # the maximum edge weight
endpoint_radius = 0.85 # minimum cosine similarity

# prob_model = generate_MLE('datasets/small_reviews/pos', 'datasets/small_reviews/neg')

emolex = generate_emolex_vectors(EMOLEX_PATH)

locations = emolex.keys()

# maximum number of words inside a graph
maximum = 14000
i = 0

# total summation of a neutral vector
neutral = 0.09999999999999999

start = time.time()

# vector for anger, joy, and sadness
G.add_node("#-ANGER")
G.add_node("#-JOY")
G.add_node("#-SAD")

endpoints = {
    "#-ANGER": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "#-JOY": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "#-SAD": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
}

k = 0

for w_i in emolex:
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

                    """
                    prob_i_j = prob_model.score(w_i, [w_j])
                    prob_j_i = prob_model.score(w_j, [w_i])

                    print(w_i + " -> " + w_j + " : " + str(prob_i_j))
                    print(w_i + " <- " + w_j + " : " + str(prob_j_i))

                    context_cost_i_to_j = 1 / (max(prob_i_j, 0.1))
                    # context_cost_j_to_i = 1 / max(prob_j_i, 0.00001)
                    """

                    if similarity_cost < cost_threshold:
                        G.add_edge(w_i, w_j, weight=similarity_cost)
                        
                    # G.add_edge(w_j, w_i, weight=1/similarity_cost)
        print("Adding Node: '" + w_i + "' - No. " + str(k) + " w. " + str(G.number_of_edges()) + " edges / " + str(time.time() - start) + " sec(s) elapsed.")
        k += 1            
        

    if i == maximum:
        break

    i += 1

"""pos = nx.spring_layout(G)

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
weights = [G[u][v]['weight'] for u,v in edges]"""

# nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
# nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
# plt.show()

# started: 4:02pm
# after 1 minute: 2500 nodes inserted
# 4:05pm: 4100 nodes inserted

nx.write_graphml(G, "graph_outputs/output_" + str(maximum) + "_wo_neutral.graphml")