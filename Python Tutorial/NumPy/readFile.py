

f= open("demo.txt","w+")

f.write("Woops! I have deleted the content!,Niger")
    
for i in range(10):
     f.write("This is line %d\r\n" % (i+3))

f.close()