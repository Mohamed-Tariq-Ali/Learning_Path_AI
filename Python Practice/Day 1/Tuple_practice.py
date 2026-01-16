tuple1 = ("tariq", 1 , 10 , "bang")
tuple2= (1, 2, 3, 4, 5 )
tup3 = "a", "b", "c", "d"

tup1=()

tup1=(60,)
print(tup1)

print("tuple1[0]:", tuple1[0])

#merging
 
tuple3 = tuple1 + tuple2
print(tuple3)

#deleting tuple


tup = ('physics', 'chemistry', 1997, 2000)

print(tup)
#del tup

#print("After deleting tup : ")
#print(tup)

# we cannot delete elements in the tuple directly so we convert it into list and remove the element then change it back to tuple

lan = list(tup)
lan.remove("physics")
tup=tuple(lan)
print(tup)