#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy
from sklearn import datasets

base = datasets.load_breast_cancer()

entries = base.data
outputs_values = base.target

outputs = numpy.empty([569, 1], dtype=int)
for i in range(569):
    outputs[i] = outputs_values[i]

weights0 = 2 * numpy.random.random((30,5)) - 1
weights1 = 2 * numpy.random.random((5,1)) - 1

epochs = 10000
learning_rate = 0.3
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
    