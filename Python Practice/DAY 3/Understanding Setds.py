my_set=set()
n=int(input("Enter the number of elements: "))
for i in range(n):
    my_set.add(int(input("Enter the element: ")))
print(my_set)

def add_element(set,element):
    set.add(element)
    print("updated set",set)

# def update_element(set,element):
#     set.update(element)

while True:
    print("\n 1.Add element 2.Update element 3.Delete element 4.Clear element 5.Union 6.difference 7.intersection "
          "8.symmetric_difference 9.issubset 10.issuperset")
    choice=int(input("Enter your choice: "))
    match choice:
        case 1:
            element = int(input("Enter the element: "))
            add_element(my_set,element)