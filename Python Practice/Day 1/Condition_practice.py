#if condition

a = 5
b=8
if a < b :
   print("B is greater than A")

# if - else condition

num = 10 
if num >=0:
   print("Positive Number")
else :
   print("Negative Number")

#if-elif-else

num = 0 
if num >0:
  print("Positive number")
elif num==0:
  print("ZERO")
else:
  print("Negative number")

#Nested-if 

num = -1 
if num>0:
   if num==0:
         print("ZERO")
   else :
        print("Positive number")
else :
   print("Negative number")


#Short Hand If statement

i=11
if i >0 :print("Positive") #one line if statement

a=3
b=4
print("A") if a>b else print("B") #one line if-else statement

#if-else statement with 3 conditions
a=3
b=6
print("A is greater than B") if a>b else print("=") if a==b else print("B is greater than A")


