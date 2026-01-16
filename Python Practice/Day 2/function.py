'''def sum(*args):
    s = 0
    for i in args:
        s += i
    print("sum is", s)

sum(1,2,3,4,5,6,7)'''

l1 = [1,2,3,4,5,6]
print(sum(l1))

def treesum(a,b,c):
    print(a,b,c)
a=[1,2,3]
treesum(*a)

def ket(a,b,c):
    print(a,b,c)
a={'a':"one",'b':"two",'c':"three"}
print(**a)