"""
Library for Vector Manipulation - John McCambridge
"""

# cosine_similarity
def similarity(vector_1, vector_2):
    return 1 - spatial.distance.cosine(vector_1, vector_2)

# euclidean_distance
def distance(vector_1, vector_2):
    return distance.euclidean(vector_1, vector_2)