f=open("demo.txt","r")
content=f.read()
print(content)
wa=open("demo.txt","w")
wn=open("demo.txt","r")

wa.write("Nalla irukingala")
print(wn.read())

add_text = open("demo.txt","a")
add_text.write("\n aprm vaalkai epdi pothu")
print(add_text)
add_text.close()

f=open("demo.txt","r")
data_read=f.read()
length = len(data_read)
print(length)

f=open("demo.txt","w")
f.write("Bankai : zanka no tachi")
f.write("\n Bankai : Tensa Zangetsu")
f.write("\n Bankai : Senbonzakura Kageyoishi")

f=open("demo.txt","r")
print(f.readline())
print(f.readline())
print(f.readline())

f.close()