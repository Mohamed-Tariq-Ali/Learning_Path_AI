'''my_list = ["banana","apple","cherry"]

my_list[0]="orange"

list1 = ["i am " , 18 , "years old"]
list1.append("HELLO")
my_list.insert(1,10)
list1.insert(2,False)

print(list1[::-1])

my_list.extend(list1)

list2 = [10,20,20,40,10,3,4]
list2.sort()
print(list2)

#list1.remove(18)

list1.reverse()
del list1 # used to delete the entire list

print(list1) # gives error since the list is deleted

list3 = list2[2:6]
list3.sort()
print(list3)




#print(list1)





#print(type(list1))
#print(len(list1))'''

#newlist = [expression for item in iterable if condition == True] syntax for list comprehension

fruits = ["avacado" , "apple", "banana" , "cherry" , "orange", "papaya","pineapple","pumpkin"]
numbers = [10,2,30,40,19,34,56,67,86]


sample = [2 ** x for x in range(10)]
print(sample)

sample2 = [x.upper() for x in fruits if "a" in x]
print(sample2)

sample3 = [x for x in numbers if x<70]
print(sample3)

sample4 = [ x if x!="banana" else "orange" for x in fruits]
print(sample4)

list5 = list(("apple","guava",12,14,False)) #list constructor , double braces must
print(list5)


