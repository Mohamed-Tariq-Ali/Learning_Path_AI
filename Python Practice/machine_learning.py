# import numpy as np
#
# n1=np.array([1,2,23,4,6])
# print(n1)
# n2=np.array([[1,2,23,4,6],[2,3,4,6,8]]) #both arrays should contain same no of elements , otherwise it wont work
#
# print(n2)
#
# n1=np.zeros((2,3))
# print(n1)
# n1=np.full((2,3),10)
# print(n1)
#
# n1=np.arange(10,20)
# print(n1)
# n1=np.arange(10,50,5)
# print(n1)
#
# n1=np.random.randint(1,100,5)
# print(n1)
#
# n1=np.array([[1,2,3], [4,5,6]])
# print(np.shape(n1))
#
# n1=np.array([10,20,30,40,50])
# n2=np.array([40,50,60,70,80])
# print(np.vstack((n1,n2)))
# print(np.hstack((n1,n2)))
# print(np.column_stack((n1,n2)))
# print(np.intersect1d(n1,n2))
# print(np.setdiff1d(n1,n2))
# print(np.setdiff1d(n2,n1))
#
# n1=np.array([10,20])
# n2=np.array([30,40])
# print(np.sum([n1,n2]))
# print(np.sum([n1,n2],axis=0))
# print(np.sum([n1,n2],axis=1))
#
# n1=n1+1
# print(n1)
# n2=n2-2
# print(n2)
# n2=n2/2
# print(n2)
# n2=n2*2
# print(n2)
#
# print(np.mean(n1))
# print(np.median(n2))
# print(np.std(n1))
#
# n1=np.array([[1,2,3],[4,5,6],[7,8,9]])
# n2=np.array([[11,22,33],[44,55,66],[77,88,99]])
#
# print(n1)
# print(n1[0])
# print(n1[:,1])
# print(n1[:,2])
#
# print(n1.transpose())
# print(n1.dot(n2))
# print(n2.dot(n1))
#
# n1=np.array([10,20,30])
# np.save('my_numpy',n1)
# n2=np.load('my_numpy.npy')
# print(n2)

#==============================PANDAS=============================

import pandas as pd
n1=pd.Series([1,2,3,4,5])
print(n1)
print(type(n1))

n1=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print(n1)

n1=pd.Series({'a':1,'b':2,'c':3,'d':4,'e':5})
print(n1)

n1=pd.Series({'a':10,'b':20,'c':30},index=['b','c','d','a'])
print(n1)

n2=pd.Series(n1,index=['d','a','b','c'])
print(n2)
print("========================================================")
n1=pd.Series([1,2,3,4,5,6,7,8,9])
print(n1[3])
print(n1[:4])
print(n1[-4:])
print("=================================================================")

print("ADDING SCALAR VALUE TO THE SERIES AND ADDING TWO SERIES")

s1=pd.Series([1,2,3,4,5,6,7,8,9])
s2=pd.Series([10,20,30,40,50,60,70,80,90])
print(s1+5)
print(s1+s2)
print(s1-s2)
print(s1*s2)
print(s1/s2)
print(s1/s1)
print("==================== DATAFRAME==================")

p1=pd.DataFrame({"Name":["Bob","sam","james"],"Marks":[76,34,78]})
print(p1)

p2=pd.read_csv('Iris.csv')
print(p2.head())
print(p2.tail())
print(p2.describe())
print(p2.shape)

print("==================ACCESSSING ROWS AND COLUMNS THROUGH ILOC[]")

print(p2.iloc[0:3,0:2])

print("==================ACCESSSING ROWS AND COLUMNS THROUGH LOC[]")

print(p2.loc[0:3,("sepal_length","sepal_width")]) #accessing based on column names


