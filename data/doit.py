f = open("ClearTest.txt", "w+")
#note that images 8349 are missing...
for i in range(8000, 8972 ):
    for j in range(0, 35):
        f.write("%04d.jpg 0\n" % (i))

f.close()
