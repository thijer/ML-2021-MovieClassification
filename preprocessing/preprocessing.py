import pandas as pd
import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import resize
import time
import signal


x = 256
# Set this to some useful value depending on remaining 
# dataset size and resulting dimension size of img
y = 320

# Start at this row of the dataset,
start = 700
# and stop here.
stop = 0

# Enable early program termination in an orderly fashion, 
# by setting a flag on keyboard interrupt (Ctrl-C).
Interrupted = False
def catch_keyboardinterrupt(signum, frame):
    global Interrupted
    Interrupted = True


def main():
    global start, stop
    signal.signal(signal.SIGINT, catch_keyboardinterrupt)
    try:
        data = pd.read_csv("data\\41K_processed.csv", header = 0)
    except FileNotFoundError:
        print("41K_processed.csv not found in data directory. Exiting.")
        return
    
    # Filter empty entries and rows smaller than y from the data.
    filtered = data[(data["rows"] >= y)]
    
    if(stop == 0) : stop = len(filtered)

    start = min(len(filtered), start)
    stop = min(len(filtered), stop)
    
    timestamp = time.time()
    for i in range(start, stop):
        try:
            img = imread(filtered.at[i, "poster"])
            img = resize(img, (y,x), preserve_range= True)
            imgname = "data\\normal\\{}.jpg".format(filtered.at[i, "id"])
            img = img.astype(np.uint8)
            imsave(imgname, img)
        except Exception as ex:
            print("Exception at {}: {}".format(i, ex))
        
        if(i % 100 == 99):
            print("Step {}/{}".format(i + 1, stop), "Duration (100 imgs):", time.time() - timestamp)
            timestamp = time.time()
            
        if(Interrupted):
            print("Stopping at", i)
            break
    print("Stopping")
    return

if __name__ == '__main__':
    main()