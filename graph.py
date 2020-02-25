"""
Algorithm for generating bidirected graph relating words with sentiment and context.
""" 
import random
import networkx as nx
import matplotlib.pyplot as plt
from numpy import loadtxt
from gensim.models import Word2Vec
from nltk.corpus import brown
from utilities.vector import distance, similarity
from utilities.dataHandler import get_corpus, generate_vectors, generate_MLE

from nltk.lm import MLE, NgramCounter
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

CORPUS_LOCATION = "datasets/corpus/cornell_small.txt"
MODEL_PATH = "models/BrownWord2Vec.model"
VECTOR_PATH = "datasets/weights/cornell_small_vectors.csv"


# brown_corpus = brown.sents()
model = Word2Vec.load(MODEL_PATH) # Word2Vec(brown_corpus)
# returns a list of uncategorized words in the corpus
corpus = get_corpus(CORPUS_LOCATION)
vectors = loadtxt(VECTOR_PATH, delimiter=',')

G = nx.Graph()

# word to vec hash map
w2v = dict()

epsilon = 2.0
probability_scaler = 1.0

for k in range(0, len(vectors)):
    word = corpus[k]
    vector = vectors[k]

    w2v[word] = vector

prob_model = generate_MLE('datasets/small_reviews/pos', 'datasets/small_reviews/neg')
options = {
    'node_color': 'blue',
    'node_size': 100,
    'width': 1
}

for i in range(0, 50):
    w_i = corpus[i]
    if w_i in model:
        G.add_node(w_i)
        print("Adding node: " + w_i)
        for w_j in list(G.nodes):
            if w_j in model and w_i != w_j:
                # connect w_i and w_j together with edge weight of similarity
                # G.add_node(w_j)
                if distance(w2v[w_i], w2v[w_j]) < epsilon:
                    similarity_cost = 1 - similarity(w2v[w_i], w2v[w_j]);

                    prob_i_j = prob_model.score(w_i, [w_j])
                    # prob_j_i = prob_model.score(w_j, [w_i])

                    context_cost_i_to_j = 1 / (probability_scaler * max(prob_i_j, 0.00001))
                    # context_cost_j_to_i = 1 / max(prob_j_i, 0.00001)

                    #if prob_i_j != 0:
                    colors = ['r','g','b']

                    G.add_edge(w_i, w_j, color=colors[random.randrange(0, 3)], weight=similarity_cost + context_cost_i_to_j)
                    # G.add_edge(w_j, w_i, weight=1/similarity_cost)

pos = nx.spring_layout(G)

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
weights = [G[u][v]['weight'] for u,v in edges]

nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
plt.show()