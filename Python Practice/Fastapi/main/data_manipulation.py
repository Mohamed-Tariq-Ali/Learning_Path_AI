
n = int(input("Enter number of elements: "))
my_list = []

while len(my_list) < n:
    value = input("Enter a number: ")

    try:
        if '.' in value:
            my_list.append(float(value))
        else:
            my_list.append(int(value))
    except ValueError:
        print("❌ Error: Please enter a valid number (int or float). Try again.")

print(my_list)


def find_max(lst):
    print(max(lst))

def find_min(lst):
    print(min(lst))

def find_sum(lst):
    print(sum(lst))

def find_avg(lst):
    print(sum(lst)/len(lst))

def sort_ascending(lst):
    return sorted(lst)
def sort_descending(lst):
    return sorted(lst,reverse=True)

while True:
    print("\n 1.Find max 2.Find min 3.Find sum 4.Find avg 5.Sort ascending 6.Sort descending 7.Exit")
    choice = int(input("Enter your choice:"))
    match choice:
        case 1:
            find_max(my_list)
        case 2 :
            find_min(my_list)
        case 3:
            find_sum(my_list)
        case 4:
            find_avg(my_list)
        case 5:
           print( sort_ascending(my_list))
        case 6:
          print(sort_descending(my_list))
        case 7:
            print("Exiting...")
            break
        case _:
            print("Invalid Choice")
