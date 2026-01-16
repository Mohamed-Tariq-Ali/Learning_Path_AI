from unittest import case

n=int(input("Enter number of elements:"))
my_lst =[]
for i in range(n):
    my_lst.append(int(input()))
print(my_lst)

def add_element(lst,element):
    lst.append(element)
    print("Updated List",lst)
def insert_element(lst,element):
    lst.insert(0,element)
    print("Updated List",lst)

def remove_element(lst,element):
    lst.remove(element)
    print("Updated List",lst)

def clear_list(lst):
    lst.clear()
    print("Updated List",lst)
def add_new_list():
    elements=input("Enter element:").split(',')
    new_listt=[int(x) for x in elements]
    print("New List",new_listt)
    return new_listt

def extend_list(lst):
    elements=input("Enter element:").split(',')
    lst.extend([int(x) for x in elements])
    print("Updated List",lst)

def sort(lst):
    lst.sort()
    print("Sorted List",lst)

while True:
    print("\n.1.Add elements 2.Insert elements 3.Remove elements 4.Clear 5.Add List 6.Extend 7.Sort 8.Exit")
    choice = int(input("Enter choice:"))
    match choice:
        case 1:
            element = int(input("Enter element you want to add:"))
            add_element(my_lst,element)
        case 2:
            element = int(input("Enter element you want to insert:"))
            insert_element(my_lst,element)
        case 3:
            element = int(input("Enter element you want to remove:"))
            remove_element(my_lst,element)
        case 4:
            clear_list(my_lst)
        case 5:
            add_new_list()
        case 6:
            extend_list(my_lst)
        case 7:
            sort(my_lst)
        case 8:
            print("Exiting")
            break





