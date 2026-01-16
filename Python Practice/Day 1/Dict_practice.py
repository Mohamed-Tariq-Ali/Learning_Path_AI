my_dict = {'first' : 'Tariq', 'Second':'Joker','age':21}
print(my_dict)

print(my_dict['age'])

dict1 = dict(Name="Tariq" , Age=21 , Location="Chennai") # even if i give without enclosing '' it converts defaultly and changes = into :
#print(dict1)

dict1['new']="value"
print(dict1)

my_dict.pop('age') # removes the element , we can access the elements only using the keys.
print(my_dict)

del dict1['new'] # deletes  the value at the specified key.
print(dict1)

sample = {x : x*x for x in range(6)}
print(sample)

dict2 = {'apple':'sour','banana':'sweet'}

res=my_dict.update(dict2)
print(my_dict)
print(res)