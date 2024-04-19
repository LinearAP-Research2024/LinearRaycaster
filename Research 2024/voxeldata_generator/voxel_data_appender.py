f = open("voxeldata.txt","a")
strr = ""
for i in range(512 * 512 * 511):
    strr += ("0\n")
f.write(strr)
print("g")
f.close()