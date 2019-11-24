# -*- coding: utf-8 -*-
import numpy

entries = numpy.array([1, 7, 5])
weights = numpy.array([0.8, 0.1, 0])

def amount(entries, weights):
    return entries.dot(weights)

def step(amount):
    if ( amount >= 1 ):
        return 1
    return 0
    
total = amount(entries, weights)
result = step(total)