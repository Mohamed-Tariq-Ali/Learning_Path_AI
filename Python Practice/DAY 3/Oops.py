# class Vehicle:
#     def __init__(self,name,mileage):
#         self.name=name
#         self.mileage=mileage
#
#     def display(self):
#         print("Name of the vehicle is:",self.name)
#         print("Mileage of the vehicle is ",self.mileage)
#
# v1 = Vehicle('Audi',140)
# v1.display()

# class Car(Vehicle):
#     def __init__(self, name, mileage, hp, tyres):
#         super().__init__(name, mileage)   # call parent constructor
#         self.hp = hp
#         self.tyres = tyres
#
#     def show_car(self):
#         print("The car goes like swiiiiiii")
#     def show_car_details(self):
#         print("no of tyres ",self.tyres)
#         print("hp",self.hp)
#
# c1 = Car('Benz',200,400,4)
# c1.show_car()
# c1.display()
# c1.show_car_details()
#
# class Dad:
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#
#     def display(self):
#         print(self.name)
#         print(self.age)
#     def beating(self):
#         print("he throws things")
#
# class Mom:
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#
#     # def display(self):
#     #     print(self.name)
#     #     print(self.age)
#     def shouting(self):
#         print("SHe scolds a lot",self.name)
#
# class Son(Dad,Mom):
#     def __init__(self,name,age):
#         Dad.__init__(self,name,age)
#         Mom.__init__(self,name,age)
#         # def display(self):
#         # print(self.name)
#         # print(self.age)
#
#
# class Driver:
#     s1=Son("Tariq",18)
#     s2=Mom("alexa",56)
#     s2.shouting()
#     s1.display()
#     s1.beating()
#     s1.shouting()
 #========MULTI LEVEL INHERITANCE============

# class Parent:
#      def assign_name(self,name):
#          self.name=name
#      def show_name(self):
#          return self.name
# class Child(Parent):
#     def assign_age(self,age):
#         self.age=age
#     def show_age(self):
#         return self.age
# class Grandchild(Child):
#     def gender(self,gender):
#         self.gender=gender
#     def show_gender(self):
#         return self.gender
#
# class Driver:
#     g1 = Grandchild()
#     g1.assign_name("Pan")
#     g1.assign_age(20)
#     g1.gender("Male")
#     print(g1.show_name())
#     print(g1.show_age())
#     print(g1.show_gender())

#=============Polymorphism=========
class Animal:
    def sound(self):
        print("the animal makes sound")
class Dog(Animal):
    def sound(self):
        print("the dog makes woof woof")
class Cat(Animal):
    def sound(self):
        print("the cat makes meow")
animals=[Dog(),Cat()]
for x in animals:
    x.sound()
