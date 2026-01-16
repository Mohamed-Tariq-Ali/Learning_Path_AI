'''import pandas as pd

data = pd.read_csv(r'D:\Analance\datasets_AAA\hr.csv')'''





'''fruits = ["apple","banana","cherry"]
for x in fruits:
   print(x)

for x in "banana":
   print(x)

for x in fruits:
   print(x)
   if x == "banana":
       break
   elif x=="apple":
       print("Obtained value")
   else :
       print("none")

for x in range(2,6):
   print(x)

#else in for loop

for x in range(5):
   print(x)
else :
   print("FINISHED")

#nested loops
fruits = ["apple","banana","cherry"]
taste = ["sour","sweet","bitter"]
for x in fruits:
   for y in taste:
       print(x,y)'''
   


#===============PROGRAM=============

'''minimum = int(input("Enter the MInimum Value :"))
maximum = int(input("Enter the Maximum Value:"))

print("Palindrome Numbers between %d" % minimum + "and %d  : " % maximum)

for num in range(minimum , maximum+1):
   temp = num
   reverse = 0
   
   while temp > 0 :
       rem = temp % 10
       reverse = (reverse * 10)+rem
       temp=temp//10

   if num == reverse :
         print("%d " % num, end='  ')'''

#================================================

rows = 5
for i in range(0,rows):
    for j in range(0,i+1):
         print("*",end=' ')
    print("\r")

for i in range(rows, 0, -1):
    for j in range(0, i - 1):
        print("*",end=' ')
    print("\r")


# square number printing

'''n = 5
for x in range(1, n):
    num = (x * x)
    print(" " + str(num), end=" ")

# pattern printing

for x in range(1, 6):
    for y in range(65, 70):
        print(chr(y), end=" ")
    print()'''

# iterating data

'''for i in data.columns:
    print(i)

# Dual iteration

for i in data.columns:
    for j in data.columns:
        print(i, j)

for i in data['SalaryType'].value_counts():
    print(i, data['SalaryType'][i].index)'''






