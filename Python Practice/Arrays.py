# print("enter the no of elements you want to add")
#
# size = int(input())
# arr=[]
# for i in range(size):
#     arr.append(int(input()))
#     print(arr)

#====================STACK USING LIST==================
'''stack=[]
stack.append("hi")
stack.append(1)
stack.append("joker")
stack.pop()
print(stack)'''
# from collections import deque
#
# #====================STACK USING COLLECTION.DEQUE
# #   DIFFERENCE BETWEEN LIST AND DEQUE IS , DEQUEUE IS FASTER
# stack=deque()
# stack.append("hi")
# stack.append("joker")
#
# print(stack.pop())
# print(stack)

#================Stack using queue=======
# from queue import LifoQueue, Queue
#
# stack=LifoQueue(maxsize=3)
# stack.put(2)
# stack.put(3)
# stack.put(4)
# print(stack.qsize())
# print(stack.full())
# temp = []
#
# while not stack.empty():
#     temp.append(stack.get())
#
# # temp now has popped values
# for x in temp:
#     print(x)
#
# # restore stack
# for x in reversed(temp):
#     stack.put(x)

# #=============BASIC IMPLEMENTATION OF QUEUE ========
#
# class Queue:
#     def __init__(self):
#         self.queue=[]
#
#     def enqueue(self,item):
#         self.queue.append(item)
#
#     def dequeue(self):
#         if len(self.queue)==0:
#             return None
#         return self.queue.pop(0)
#     def display(self):
#         print(self.queue)
#     def empty(self):
#         if len(self.queue)==0:
#             return True
#         return False
#     # def empty(self):
#     #     return self.empty() and self.queue==[]
#     def peekfirst(self):
#         if len(self.queue)==0:
#             return None
#         return self.queue[0]
#     def peeklast(self):
#         if len(self.queue)==0:
#             return None
#         return self.queue[-1]
#
# q=Queue()
# q.enqueue(25)
# q.enqueue(35)
# q.enqueue(45)
# q.enqueue(55)
# q.display()
# q.dequeue()
# q.display()
# print(q.empty())
# print(q.peekfirst())
# print(q.peeklast())

# #LINKED LIST INTERNAL IMPLEMENTATION
#
# class Node:
#     def __init__(self,data):
#         self.data=data
#         self.next=None
#
# class SinglyLinkedList:
#     def __init__(self):
#         self.head=None
#
#     def insert_at_beginning(self,data):
#         new_node = Node(data) #passing the data inside the Node and storing it in new node
#         new_node.next=self.data
#         self.data=new_node.data
#
#     def insert_at_end(self,data):
#         new_node=Node(data)
#
#         if self.head is None:
#             self.head=new_node
#             return
#
#         temp=self.head
#         while(temp.next is not None):
#             temp=temp.next
#         temp.next=new_node
#     def inseert_at_position(self,position,data):
#         if position==0:
#             self.insert_at_beginning(data)
#             return

#=====================================================================================================

#LINEAR SEARCH
#
# def linear_search(arr,x,n):
#     for i in range(0,n):
#         if arr[i]==x:
#             return i
#     return -1
# arr=[1,2,3,4,5,6,7,8,9]
# x=7
# n=len(arr)
# result=linear_search(arr,x,n)
# if result==-1:
#     print("element not found")
# else:
#     print("element found at index",result)

#======================================================================
# #BINARY SEARCH
# def binary_search(arr, begin, end, target):
#     if begin > end:
#         return -1
#
#     mid = (begin + end) // 2
#
#     if arr[mid] == target:
#         return mid
#     elif arr[mid] < target:
#         return binary_search(arr, mid + 1, end, target)
#     else:
#         return binary_search(arr, begin, mid - 1, target)
#
#
# arr = [1, 3, 5, 7, 9]
# print(binary_search(arr, 0, len(arr) - 1, 7))

#===========================================================================
#INSERTION SORT

def insertion_sort(arr):
    for i in range(1,len(arr)):
        key=arr[i]
        j=i-1
        while j>=0 and arr[j]>key:
            arr[j+1]=arr[j]
            j=j-1
        arr[j+1]=key

arr = [12, 11, 13, 5, 6]
insertion_sort(arr)
print("Sorted array:", arr)











