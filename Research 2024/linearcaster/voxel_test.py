import torch
import time
import linearcaster_cpp

linearcaster_cpp.compile_voxelimage()
print("compile done")
final = []
start=time.time()
final = linearcaster_cpp.siddon_raycast(
    int(1024),
    torch.tensor([0.0,0.0,512.0]),
    torch.tensor([0.0,0.0,0.0]),
    int(512)
)
torch.cuda.synchronize()
print("64 final time: {} s".format(time.time() - start))

#"""
with open('image_list.txt', 'w+') as f:
    f.write('flattened_array = [\n')
    # write elements of list
    for items in range(0,len(final)-1):
        f.write("{},".format(final[items]))
    f.write("{}".format(final[len(final)-1]))
    print("File written successfully")
    f.write(']')
 
# close the file
f.close()
#"""