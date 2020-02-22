"""
Library for Vector Manipulation - John McCambridge
"""
from scipy import spatial
from scipy.spatial import distance
from numpy import dot
from numpy.linalg import norm

# cosine_similarity
def similarity(vector_1, vector_2):
    return dot(vector_1, vector_2)/(norm(vector_1)*norm(vector_2)) #1 - spatial.distance.cosine(vector_1, vector_2)

# euclidean_distance
def distance(vector_1, vector_2):
    return spatial.distance.euclidean(vector_1, vector_2)