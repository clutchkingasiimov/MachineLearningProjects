
"""Problem Statement

** Your firm is considering the launch of a new product, the XJ5. The upfront development cost
is $10 million, and you expect to earn a cash flow of $3 million per year for the next five years.
Plot the NPV profile for this project for discount rates ranging from 0% to 30%. For what
range of discount rates is the project attractive? ** """

import numpy as np
import matplotlib.pyplot as plt 

interest = [x/100 for x in range(1, 31)]
cost = -10_000_000
cashflow = 3_000_000

def best_npv(cost, cashflow, interest, time):
    if time is True:
        discounted_cashflow = []
        time_range = [x for x in range(1, time+1)
        for rates, time in zip(interest, time_range):
            print('--------------------Iteration--------------------')
            dcf_vals = (cashflow/(1+rates) ** times)
            discounted_cashflow.append(dcf_vals)
            """What all to do in this code
1. Make a way to store each interest rate's cash flow in a separate row of a matrix/dataframe according to the timeline
2. Calcualte the NPV for each interest rate and flag the rows which have a Positive NPV
3. Plot the change in NPV from different interest rates to get a better idea of where things start to get profitable
"""

    print('Discounted Cashflow')
    print(discounted_cashflow)
                      
