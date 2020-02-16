#Two distinct elements
"""
In each input list, every number repeats 
atleast once, except for two.

Goal: Write a function that returns two unique
numbers.

Example: 
returnUnique([1, 9, 8, 8, 7, 6, 1, 6]) ➞ [9, 7]

returnUnique([5, 5, 2, 4, 4, 4, 9, 9, 9, 1]) ➞ [2, 1]

returnUnique([9, 5, 6, 8, 7, 7, 1, 1, 1, 1, 1, 9, 8]) ➞ [5, 6]
"""
import numpy as np 
import time

def returnUnique(numbers):
    #Input an array or a list 

    shape = len(numbers)
    sim_matrix = np.zeros((shape, shape))

    for i in range(len(numbers)):
        for j in range(len(numbers)):
            #Run linear search across array 
            if numbers[i] == numbers[j]:
                sim_matrix[i][j] = 1
            else:
                sim_matrix[i][j] = 0
    
    #Check for columns where only one "1" exists (Kinda like finding the basis vectors of the matrix)
    count = [sum(sim_matrix[i]) for i in range(len(numbers))]
    unique = [numbers[i] for i in range(len(numbers)) if count[i] == 1]
    print("Unique numbers: {}".format(unique))


#Not the most optimal solution, at O(n^2) right now. 
numbs = np.random.randint(500, size=10)
returnUnique(numbs)
