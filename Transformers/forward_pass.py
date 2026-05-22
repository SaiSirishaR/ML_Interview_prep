## Forward pass

# input = x [neurons in l-1, neurons in l]
# weights in layer 1 = w1; bias in layer 1 = b1
# activation in layer 1 = a1 = w1*x +b1
# weights in layer 2 = w2; biase in layer 2 = b2   [weights = [neurons in output layer, neurons in input layer]; bias = [neurons in layer, 1]]
# activation in layer 2 = a2 - w2*a1+b2
import numpy as np
import math

def weights(i,o):
    return np.random.rand(i,o)

def bias(n):
    return np.random.rand(n,1)
    
def cal_preactivation(inp, w, b):
    a = np.dot(w, inp) + b
    return a

def act_fn(x, n):
    return cal_preactivation(x.reshape(-1,1),weights(n[1], n[0]), bias(n[1]))

    
input = [1,3,4]
nodes_per_layer = [3,2,1]
x= np.array(input)
output = act_fn(x, nodes_per_layer)
print("layer 1 output is", output, "shape is", np.shape(output))
