sample = {1,2,2,3,3,}
sample1 = set((1,2,3,3,))
sample2 = set([1,2,3,4,5,6,7,2,3,4])
print(sample2)

sample.add(6)#used to add single value

sample.update([4,7,8]) #used to add multiple values.
print(sample)

#removing

sample.remove(1)
sample.discard(2)
print(sample)

#clear all
sample.clear()
print(sample)

#set operations
x={1,2,3,4,5,6}
y={"a","b","c","d"}
#union
print(x|y)
z = x.union(y)
print(z)

#intersection - Gives only the duplicate values in both the set

a = {34,56,67}
b = {45,56,78,98}
print(a&b)
c = a.intersection(b)
print(c)


#difference - It gives the elements that are only present in the first set which are not in other set.

h = {23,34,45,56}
g = {56,45,00,67}

print(h-g)
j = h.difference(g)
print(j)

#symmetric_difference
# we can use ^ instead ,it will give the same result
t = h.symmetric_difference(g) #it gives the elements that are not common in both the sets.
print(t)
print(h^g)

#FROZEN SET - we cannot add or remove elements to the frozen set.

x=frozenset([1,2,3,4,5,6])
y = frozenset([1,2,1,2,3,4,3,3,4,4,4,5,5,5,6,6,6,7,7,7])
print(x)
print(y)

z=x.copy() #copies the elements.
z=x.union(y)
z=x.intersection(y)
z=x.difference(y)
print(z)
