import pandas as pd
import numpy as np
from skimage.io import imread
import os

path = "data\\normal\\"

def main():
    try:
        data = pd.read_csv("data\\41K_processed.csv", header = 0, index_col = "id")
    except FileNotFoundError:
        print("41K_processed.csv not found in data directory. Exiting.")
        return
    
    data["channels"] = 0
    # Filter broken poster links from the data
    # data = data[(data["rows"] != 0)]
    files = os.listdir(path)

    for imgname in files:
        index = int(imgname.split(".")[0])
        img = imread(path + imgname)
        try:
            data.at[index, "channels"] = img.shape[2]
        except IndexError:
            data.at[index, "channels"] = 1
        
    data.to_csv("data\\41K_processed_v2.csv")

if __name__ == '__main__':
    main()