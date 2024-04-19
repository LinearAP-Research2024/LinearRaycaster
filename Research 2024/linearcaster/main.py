import math
import torch
import time
# Our module!
import linearcaster_cpp
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


#full = read_file_as_string("polydata.txt")
#print(full)
linearcaster_cpp.compile_polyimage("polydata.txt")

linearcaster_cpp.compile_voxelimage()
print("compile done")

final = linearcaster_cpp.linear_raycast(
    int(64),
    torch.tensor([0.0,0.0,2.0]),
    torch.tensor([0.0,0.0,-2.0]),
    int(2)
)

final = []
tracemalloc.start()
start=time.time()
final = linearcaster_cpp.linear_raycast(
    int(1024),
    torch.tensor([0.0,0.0,2.0]),
    torch.tensor([0.0,0.0,-2.0]),
    int(2)
)
torch.cuda.synchronize()
print("Linear 1024 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

final = []
tracemalloc.start()
start=time.time()
final = linearcaster_cpp.siddon_raycast(
    int(1024),
    torch.tensor([0.0,0.0,512.0]),
    torch.tensor([0.0,0.0,0.0]),
    int(512)
)
torch.cuda.synchronize()
print("Siddon 1024 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

final = []
tracemalloc.start()
start=time.time()
final = linearcaster_cpp.linear_raycast(
    int(512),
    torch.tensor([0.0,0.0,2.0]),
    torch.tensor([0.0,0.0,-2.0]),
    int(2)
)
torch.cuda.synchronize()
print("Linear 512 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

final = []
tracemalloc.start()
start=time.time()
final = linearcaster_cpp.siddon_raycast(
    int(512),
    torch.tensor([0.0,0.0,512.0]),
    torch.tensor([0.0,0.0,0.0]),
    int(512)
)
torch.cuda.synchronize()
print("Siddon 512 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

final = []
start=time.time()
final = linearcaster_cpp.linear_raycast(
    int(256),
    torch.tensor([0.0,0.0,2.0]),
    torch.tensor([0.0,0.0,-2.0]),
    int(2)
)
torch.cuda.synchronize()
print("Linear 256 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

final = []
tracemalloc.start()
start=time.time()
final = linearcaster_cpp.siddon_raycast(
    int(256),
    torch.tensor([0.0,0.0,512.0]),
    torch.tensor([0.0,0.0,0.0]),
    int(512)
)
torch.cuda.synchronize()
print("Siddon 256 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

final = []
start=time.time()
final = linearcaster_cpp.linear_raycast(
    int(128),
    torch.tensor([0.0,0.0,2.0]),
    torch.tensor([0.0,0.0,-2.0]),
    int(2)
)
torch.cuda.synchronize()
print("128 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

final = []
tracemalloc.start()
start=time.time()
final = linearcaster_cpp.siddon_raycast(
    int(128),
    torch.tensor([0.0,0.0,512.0]),
    torch.tensor([0.0,0.0,0.0]),
    int(512)
)
torch.cuda.synchronize()
print("Siddon 128 final time: {} s".format(time.time() - start))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
#print(final)
# open file
"""
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
"""