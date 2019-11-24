# -*- coding: utf-8 -*-

entries = [1, 7, 5]
weights = [0.8, 0.1, 0]

def amount(entries, weights):
    total = 0
    for i in range(3):
        total += entries[i] * weights[i]
    return total

def step(amount):
    if ( amount >= 1 ):
        return 1
    return 0
    
total = amount(entries, weights)
result = step(total)