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
start = 0
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
        data = pd.read_csv("data\\duplicate_free_41K.csv", header = 0)
    except FileNotFoundError:
        print("csv not found in data directory. Exiting.")
        return
    
    data["rows"] = 0
    data["cols"] = 0
    data["ratio"] = 0.0
    # data["channels"] = 0

    # Filter empty entries and rows smaller than y from the data.
    # data = data[(data["rows"] >= y)]
    
    if(stop == 0) : stop = len(data)

    start = min(len(data), start)
    stop = min(len(data), stop)
    
    timestamp = time.time()
    for i in range(start, stop):
        try:
            img = imread(data.at[i, "poster"])
            
            if(len(img.shape) == 3):
                if(
                    img.shape[0] >= y and
                    img.shape[1] >= x and
                    img.shape[2] == 3
                ):
                    data.at[i, "rows"] = img.shape[0]
                    data.at[i, "cols"] = img.shape[1]
                    data.at[i, "ratio"] = img.shape[0] / img.shape[1]
                    # data.at[i, "channels"] = img.shape[2]

                    img = resize(img, (y,x), preserve_range= True)
                    imgname = "data\\normal\\{}.jpg".format(data.at[i, "id"])
                    imsave(imgname, img.astype(np.uint8))
        except Exception as ex:
            print("Exception at {}: {}".format(i, ex))
        
        if(i % 100 == 99):
            print("Step {}/{}".format(i + 1, stop), "Duration (100 imgs):", time.time() - timestamp)
            timestamp = time.time()
            data.to_csv("data\\41K_processed_v3.csv", index = False)
            
        if(Interrupted):
            print("Stopping at", i)
            break
    print("Stopping")
    data.to_csv("data\\41K_processed_v3.csv", index = False)
    return

if __name__ == '__main__':
    main()