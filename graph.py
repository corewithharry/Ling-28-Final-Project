"""
Algorithm for generating bidirected graph relating words with sentiment and context.
""" 
import networkx as nx
# from gensim.models import Word2Vec
from utilities.vector import distance, similarity
from scipy import spatial
from scipy.spatial import distance


