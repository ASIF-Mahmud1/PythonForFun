# from numpy import array
# l= [1,2,3,4,5]
# a= array(l)
# print(a)

# print(a.shape)

import numpy as np 
a = np.array([1,2,3]) 
# print (a)

A= np.array([ [1,2],[3,4],[5,6] ])
B= np.array([ [1,2],[3,4] ])
print(B)
C= A.dot(B)
# D= A@B
print("C is \n",C)