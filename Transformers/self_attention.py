# 1. compute similarity scores (dot product of q,k)
# 2. softmax (similarity scores/sqroot.dk)
# 3. self-attention = 2 * V

# (exp(q.kT/dim)/sum(exp(q.kT/dim)))

import numpy as np
import math


def similarity_scores(q,k):
    return np.dot(q,k.T)/math.sqrt(np.shape(k)[1])

def softmax(sc):
    exp_sc = np.exp(sc)
    return np.exp(sc)/np.sum(exp_sc, axis = 1, keepdims = True)

def self_attention(w, v):
    return np.dot(w,v)

      
query = np.array([[3,4], [5,6]])
keys = np.array([[3,4], [5,6]])
values = np.array([[3,4], [5,6]])

scores = similarity_scores(query, keys)
weights = softmax(scores)  
self_attenion_out = self_attention(weights, values)
self_attenion_out
