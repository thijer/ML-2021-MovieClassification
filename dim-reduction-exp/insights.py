import pandas as pd
import numpy as np
from skimage.io import imsave, imread
import time


if __name__ == '__main__':
    data = pd.read_csv("data\\duplicate_free_41K.csv", header = 0)

    rows = len(data)
    # rows = 100
    
    img_rows = np.zeros(rows)
    img_cols = np.zeros(rows)
    img_ratio = np.zeros(rows)
    data["rows"] = 0.0
    data["cols"] = 0.0
    data["ratio"] = 0.0
    
    timestamp = time.time()
    for i in range(rows):
        try:
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
        if(i % 100 == 99):
            print("Step {}/{}".format(i + 1, rows), "Duration (100 imgs):", time.time() - timestamp)
            timestamp = time.time()
            data.to_csv("data\\41K_processed.csv")
    
    imgs_rows   = img_rows[img_rows != 0]
    imgs_cols   = img_cols[img_cols != 0]
    imgs_ratio  = img_ratio[img_ratio != 0]
    print("Mean n columns: {}".format(np.mean(imgs_cols)))
    print("Std  n columns: {}".format(np.std(imgs_cols)))
    print("Max  n columns: {}".format(np.max(imgs_cols)))
    print("Min  n columns: {}".format(np.min(imgs_cols)))
    
    print("Mean n rows: {}".format(np.mean(imgs_rows)))
    print("Std  n rows: {}".format(np.std(imgs_rows)))
    print("Max  n rows: {}".format(np.max(imgs_rows)))
    print("Min  n rows: {}".format(np.min(imgs_rows)))
    
    print("error rate: {}/{}".format(len(imgs_rows), rows))
    print("rotated imgs: {}/{}".format(len(imgs_ratio[imgs_ratio < 1.0]), len(imgs_ratio)))

    data.to_csv("dim-reduction-exp\\41K_processed.csv")
    
