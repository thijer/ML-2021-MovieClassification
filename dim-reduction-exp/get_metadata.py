import pandas as pd
import numpy as np
from skimage.io import imread
import time
import signal

Interrupted = False
# Start at this row of the dataset,
start = 0
# and stop here.
stop = 3873

def catch_keyboardinterrupt(signum, frame):
    global Interrupted
    Interrupted = True


if __name__ == '__main__':
    # Register a keyboard interrupt that allow us to stop without losing progress
    signal.signal(signal.SIGINT, catch_keyboardinterrupt)
    
    try:
        # Continue where we left off.
        data = pd.read_csv("data\\41K_processed.csv", header = 0)
    except FileNotFoundError:
        # Or run it for the first time.
        try:
            data = pd.read_csv("data\\duplicate_free_41K.csv", header = 0)
            # Add columns with zeros, which will be filled in during the for loop later.
            data["rows"] = 0
            data["cols"] = 0
            data["ratio"] = 0.0
        except FileNotFoundError:
            print("No csvs found")
            exit()
    
    rows = len(data)
    # rows = 100
    if(stop == 0): stop = rows

    img_rows = np.zeros(rows)
    img_cols = np.zeros(rows)
    img_ratio = np.zeros(rows)
    
    
    timestamp = time.time()
    for i in range(start, stop):
        try:
            # Read image from URL, and store it in img as a numpy ndarray
            img = imread(data.iloc[i]["poster"])
            data.at[i, "rows"] = img.shape[0]
            data.at[i, "cols"] = img.shape[1]
            data.at[i, "ratio"] = img.shape[0] / img.shape[1]
            
            img_rows[i] = img.shape[0]
            img_cols[i] = img.shape[1]
            img_ratio[i] = img.shape[0] / img.shape[1]
            # print("Step:", i)
        except:
            # print("Step:", i, "failed")
            pass
        # Provide report to standard output to let us know that it is doing something.
        if(i % 100 == 99):
            print("Step {}/{}".format(i + 1, stop), "Duration (100 imgs):", time.time() - timestamp)
            timestamp = time.time()
            data.to_csv("data\\41K_processed.csv")
        if(Interrupted):
            print("Stopping at", i)
            data.to_csv("data\\41K_processed.csv")
            exit()