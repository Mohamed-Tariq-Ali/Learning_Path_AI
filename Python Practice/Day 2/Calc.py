a=int(input("Enter your num1:"))
b=int(input("Enter your num2:"))

def add(a,b):
    return a+b
def sub(a,b):
    return a-b
def mul(a,b):
    return a*b
def div(a,b):
    return a/b
def mod(a,b):
    return a%b
def pow(a,b):
    return a**b

while True:
    print("\n 1.Add 2.Subtract 3.Divide 4.Mod 5.pow 6.Multiply 7.Exit ")
    choice = int(input("Enter your choice:"))

    match choice :
        case 1:
         print(f"The addition of {a} and {b}  is:",add(a,b))

        case 3:
            try:
                print(f"Division of {a} and {b} is",div(a,b))

            except ZeroDivisionError:
                print("Invalid input, please try again")
                break

        case 2 :
            print(f"the subtraction of {a} and {b} is",sub(a,b))
        case 4:
            print(f"the mod of {a} and {b} is",mod(a,b))
        case 5:
            print(f"the power of {a} and {b} is",pow(a,b))
        case 6:
            print(f"the multiplication of {a} and {b} is",mul(a,b))
        case 7 :
            print("Exiting.....")
            break
