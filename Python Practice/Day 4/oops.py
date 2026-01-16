#
#
# class Dog:
#     def __init__(self,name):
#         self.name=name
#         self.age=17
#     def bark(self):
#         print(self.name,"wowowowoow")
#
# my_dog = Dog('Bankai')
# my_dog.bark()

# class Animal:
#     def __init__(self,name,legs):
#         self.name=name
#         self.legs=legs
#
# class Dog(Animal):
#     def sound(self):
#         print("woof")
#
# my_dog=Dog("Jinx",5)
# my_dog.sound()
# print(my_dog.name,my_dog.legs)
class Employee:
    def __init__(self,name,age,salary,gender):
        self.name = name
        self.age=age
        self.salary=salary
        self.gender=gender

    def display(self):
        print("Employee name is:",self.name)
        print("Employee age is:",self.age)
        print("Employee salary is:",self.salary)
        print("Employee gender is:",self.gender)

e1 = Employee("Sam",32,12000,"Male")
e1.display()
