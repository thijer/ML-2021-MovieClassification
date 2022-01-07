from sklearn.decomposition import IncrementalPCA
from skimage.io import imsave, imread
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# import asyncio

def main(path: str, components: int, batch_size: int):
    
    data = pd.read_csv(path, header = 0)

    rows = len(data)
    batches = int(np.floor(rows / batch_size))
    print("Rows: {}\tBatches: {}".format(rows, batches))

    # poster = imread(image_url)
    # poster = np.reshape(poster, (1, poster.shape[0] * poster.shape[1] * poster.shape[2]))
    pca = IncrementalPCA(components, batch_size = batch_size)

    imgs = []
    for i in range(0, batches * batch_size):
        ii = i % batch_size
        imgs.append(imread(data.iloc[i]["poster"]).flatten())
        if(ii == batch_size - 1):
            pca.partial_fit(imgs)

    plt.grid()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.savefig('Scree plot.png')


if __name__ == '__main__':
    """ 
        Use Principal component analysis to reduce the number of dimensions on an image.

        Args:
        ----------
            file : int
                number of the image file

            components : int
                Number of dimensions to reduce to.
            
    """
    
    # if(len(sys.argv) != 3):
    #     exit()
    # components = sys.argv[2]
    components = 1024
    # path = os.getcwd() + "/data/duplicate_free_41K.csv"
    main("data\\duplicate_free_41K.csv", components, 100)