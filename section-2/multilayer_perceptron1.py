# -*- coding: utf-8 -*-

import numpy

entries = numpy.array([[0,0],[0,1],[1,0],[1,1]])
outputs = numpy.array([[0],[1],[1],[0]])
weights0 = numpy.array([[-0.424, -0.740, -0.961], [0.358, -0.577, -0.469]])
weights1 = numpy.array([[-0.017],[-0.893],[0.148]])

#weights0 = 2 * numpy.random.random((2,3)) - 1
#weights1 = 2 * numpy.random.random((3,1)) - 1

epochs = 100000
learning_rate = 10
momentum = 1

def sigmoid(x):
    return 1 / ( 1 + numpy.exp(-x) )

def sigmoid_derivative(x):
    return x * ( 1 - x )

for j in range(epochs):
    entry_layer = entries
    
    synaptic_sum0 = numpy.dot(entry_layer, weights0)
    hidden_layer = sigmoid(synaptic_sum0)
    
    synaptic_sum1 = numpy.dot(hidden_layer, weights1)
    result = sigmoid(synaptic_sum1)
    
    error = outputs - result
    
    absolute_error = abs(error)
    absolute_mean = absolute_error.mean()
    print("Erro: " + str(absolute_mean))
    
    output_derivative = sigmoid_derivative(result)
    output_delta = error * output_derivative
    
    weights1_transpose = weights1.T
    output_delta_prod_weight = output_delta.dot(weights1_transpose)
    delta_hidden_layer = output_delta_prod_weight * sigmoid_derivative(hidden_layer)
    
    hidden_layer_transpose = hidden_layer.T
    new_weights1 = hidden_layer_transpose.dot(output_delta)    
    weights1 = ( weights1 * momentum ) + ( new_weights1 * learning_rate )
    
    entry_layer_transpose = entry_layer.T
    new_weights0 = entry_layer_transpose.dot(delta_hidden_layer)
    weights0 = ( weights0 * momentum ) + ( new_weights0 * learning_rate )
    

# weight1_0
# -13.4554  -1.02607	  6.15912
# 6.16094   -1.02596	  -13.4509

# weights1
# 18.5575
# -48.2223
# 18.5565

