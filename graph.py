"""
Algorithm for generating bidirected graph relating words with sentiment and context.
""" 
import networkx as nx
import matplotlib.pyplot as plt
from numpy import loadtxt
from gensim.models import Word2Vec
from nltk.corpus import brown
from utilities.vector import distance, similarity
from utilities.dataHandler import get_corpus, generate_vectors

CORPUS_LOCATION = "datasets/corpus/cornell_small.txt"
MODEL_PATH = "models/BrownWord2Vec.model"
VECTOR_PATH = "datasets/weights/cornell_small_vectors.csv"


# brown_corpus = brown.sents()
model = Word2Vec.load(MODEL_PATH) # Word2Vec(brown_corpus)
# returns a list of uncategorized words in the corpus
corpus = get_corpus(CORPUS_LOCATION)
vectors = loadtxt(VECTOR_PATH, delimiter=',')

G = nx.Graph()

# add initial node to graph
# G.add_node(corpus[0])

# word to vec hash map

w2v = dict()

for k in range(0, len(vectors)):
    word = corpus[k]
    vector = vectors[k]

    w2v[word] = vector

for i in range(0, 50):
    w_i = corpus[i]
    if w_i in model:
        G.add_node(w_i)
        for w_j in list(G.nodes):
            if w_j in model:
                # connect w_i and w_j together with edge weight of similarity
                G.add_node(w_j)
                G.add_edge(w_i, w_j, weight=1 - similarity(w2v[w_i], w2v[w_j]))

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

"""word_1 = 'dismal'
word_2 = 'cry'

vector_1 = model.wv.word_vec(word_1)
vector_2 = model.wv.word_vec(word_2)

def sim(w1,w2,model):
    A = model[w1]; B = model[w2]
    return sum(A*B)/((pow(sum(pow(A,2)),0.5)*pow(sum(pow(B,2)),0.5)))


print(sim(word_1, word_2, model))
print(similarity(vector_1, vector_2))
print(model.wv.similarity(word_1, word_2))"""
