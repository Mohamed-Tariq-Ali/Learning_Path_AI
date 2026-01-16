#creating List
my_list = [1,3,"name",10.7,2,4]

#Slicing List
print(my_list[2])
print(my_list[2:5])
print(my_list[:])  #returns entire list
print(my_list[::-1] )   #returns reverse of list

#Common List Methods
mylist1= [32,11,10.0, "python", "data"]
print(mylist1)
#append
mylist1.append(21)
print(mylist1)

#insert
mylist1.insert(3,"name")
print(mylist1)

#extend
newlist= [30,20,40,10,20]
mylist1.extend(newlist)

print(mylist1)

mylist1.append(newlist)
print(mylist1)

#sort

#mylist1.sort() #it throws error ,because of presence of string variables

list=mylist1[6:9]
print(list)

list.sort()

print(list)

#remove

mylist1.remove("data")

#reverse

list.reverse()

print(list)

#deleting a list

del list

#list comprehension

sample = [2 ** x for x in range(10)]
print(sample)




