# import pandas as pd
# s = pd.Series()
# print(s)

# import numpy as numpy
# import pandas as pd

# pd.read_csv('zoo1.csv', delimiter = ',')

import pandas as pd
import numpy as np

# Series
# 
# pandas.Series( data, index, dtype, copy)
#  
# data = np.array(['a','b','c','d'])
# s = pd.Series(data)
# print (s)

# data = {'a' : 0., 'b' : 1., 'c' : 2.}
# s = pd.Series(data)
# print s

# s = pd.Series(5, index=[0, 1, 2, 3])
# print s

# data = {'a' : 0., 'b' : 1., 'c' : 2.}
# s = pd.Series(data,index=['b','c','d','a'])
# print s

# s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve the first element
# print s[0]

# s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

# #retrieve the first three element
# print( s[:3])

s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve multiple elements
print s[['a','c','d']]