#Stack and Queue
from collections import deque

#Implementing a stack 

counter = [i for i in range(10)]

stack = deque(counter)

"""
A Stack follows a LIFO array sorting. Hence for the stack, 
the last variable inserted will be removed first from the stack.
"""

for i in range(len(counter)):
   print(stack.pop())
    

"""
A Queue follows a FIFO array sorting. Hence for the queue, 
the first variable inserted will be removed from the queue
"""
queue = deque(counter)

for i in range(len(counter)):
    print(queue.popleft())

