import pandas as pd
# import numpy as np
# from skimage.io import imsave, imread

if __name__ == '__main__':
    try:
        data = pd.read_csv("data\\41K_processed.csv", header = 0)
    except FileNotFoundError:
        print("41K_processed.csv not found in data directory. Exiting.")
        exit()
    
    processed = data[data["cols"] != 0]

    print("Mean n columns: {}".format(processed["cols"].mean()))
    print("Std  n columns: {}".format(processed["cols"].std()))
    print("Max  n columns: {}".format(processed["cols"].max()))
    print("Min  n columns: {}".format(processed["cols"].min()))
    
    print("Mean n rows: {}".format(processed["rows"].mean()))
    print("Std  n rows: {}".format(processed["rows"].std()))
    print("Max  n rows: {}".format(processed["rows"].max()))
    print("Min  n rows: {}".format(processed["rows"].min()))
    
    print("error rate: {}/{}".format(len(data) - len(processed), len(data)))
    print("rotated imgs: {}/{}".format(len(processed[processed["ratio"] < 1.0]), len(processed)))

    
