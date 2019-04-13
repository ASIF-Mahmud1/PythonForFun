import pandas as pd

# pandas.DataFrame( data, index, columns, dtype, copy)

data = [1,2,3,4,9]
df = pd.DataFrame(data,index=['a','b','c','d','e'])
# print df

thoughtProcess= [ ['Year: 1', "Pissed Off"], ['Year: 2', 'Disinterested'], ["Year: 3", 'Focus on other activities'], ["Year: 4", "Reincarnation!. Triggered to learn"] ]

showThisInTable= pd.DataFrame(thoughtProcess, columns=['Year', 'State of Mind'])
print showThisInTable

# Another way to do the same thing as above

data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print df

data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
print df