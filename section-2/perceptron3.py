# -*- coding: utf-8 -*-

import numpy

entries = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = numpy.array([0, 0, 0, 1])
weights = numpy.array([0.0, 0.0])
learning_rate = 0.1

def step(total):
    if ( total >= 1):
        return 1
    return 0

def calculate_output(entries):
    total = entries.dot(weights)
    return step(total)

def train():
    totalError = 1
    while ( totalError != 0 ):
        totalError = 0
        for i in range(len(outputs)):
            calculated_output = calculate_output(numpy.array(entries[i]))
            error = abs(outputs[i] - calculated_output)
            totalError = error
            for j in range(len(weights)):
                weights[j] = weights[j] + (learning_rate * entries[i][j] * error)
                print('Weight updated: ' + str(weights[j]))
            
train()
print('\nTrained neural network!')