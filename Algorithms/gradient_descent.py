#Linear Regression simulation with estimation of OLS coefficients 

import numpy as np
import matplotlib.pyplot as plt 

#Generate regression data 
numbers = np.random.normal(5, 1, 200)
epsilon = np.random.normal(0, 1, 200)
b1 = 0.35
b2 = 0.60

y = [b1+ b2*numbers[i] + epsilon[i] for i in range(len(numbers))]

#Perform gradient descent 
b1_hat = 0 
b2_hat = 0
for i in range(len(numbers)):
    b2_hat += -2*numbers[i] * (y[i] - (b2*numbers[i]+ b1))
    b1_hat += -2*(y[i] - (b2*numbers[i] + b1))

b1 -= (b1_hat/len(numbers)) * 0.0001
b2 -= (b2_hat/len(numbers)) * 0.0001
print(b1)
print(b2)

#Plot the new regression slope with the computed parameters
y_new = [b1 + b2*numbers[i] for i in range(len(numbers))]
plt.scatter(numbers, y, edgecolor='black', color="#00d0ff")
plt.plot(numbers, y_new, color='red')
plt.title("Gradient Descent Linear Regression")
plt.show()
