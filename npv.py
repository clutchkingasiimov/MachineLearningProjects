
"""Problem Statement

** Your firm is considering the launch of a new product, the XJ5. The upfront development cost
is $10 million, and you expect to earn a cash flow of $3 million per year for the next five years.
Plot the NPV profile for this project for discount rates ranging from 0% to 30%. For what
range of discount rates is the project attractive? ** """

"""What all to do in this code
1. Make a way to store each interest rate's cash flow in a separate row of a matrix/dataframe according to the timeline
2. Calcualte the NPV for each interest rate and flag the rows which have a Positive NPV
3. Plot the change in NPV from different interest rates to get a better idea of where things start to get profitable
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

interest = [x/100 for x in range(1, 31)]
time = [x for x in range(1, 6)]
cost = 10_000_000
cashflow = 3_000_000

dcf_list = []
for rate in interest:
    for times in time:
        values = (cashflow/(1+rate) ** times)
        dcf_list.append(values)

#Reshaping the array 
dcf_list = np.array(dcf_list)
dcf_list = dcf_list.reshape(30, 5)
#Finding the sum of the total cashinflow 
cash_inflow = []
for vals in range(len(interest)):
    pvsum_inflows = np.sum(dcf_list[vals, :])
    cash_inflow.append(pvsum_inflows)
print(len(cash_inflow))
print(cash_inflow[0])
#Plot to see the relation of NPV and interest rate
plt.plot(interest, cash_inflow, '-o', color='red')
plt.xlabel('Interest rates')
plt.ylabel('PV of cashflows')
plt.show()
#At what interest rate is NPV becoming negative
npv_df = pd.DataFrame(cash_inflow, interest)
boolean = []
for vals in range(len(interest)):
    statement = cost > cash_inflow[vals]
    boolean.append(statement)

boolean = [str(i) for i in boolean]
npv_df = pd.DataFrame([cash_inflow, interest, boolean]).T
npv_df.to_csv('npv_df')

