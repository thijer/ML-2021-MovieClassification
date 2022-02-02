from operator import index
from sklearn.decomposition import IncrementalPCA
from skimage.io import imsave, imread
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

img_dim = 320 * 256 * 3
path = "data/"
batch_size = 2000
components = 100

def load_img(id):
    return imread(path + "normal/{}.jpg".format(id[0])).flatten()

def main():
    
    data = pd.read_csv(path + "41K_processed_v3.csv", index_col='id')
    data = data[data["cols"] != 0]

    rows = len(data)
    # rows = 300
    batches = int(np.floor(rows / batch_size))
    rest = rows - batches * batch_size

    print("Rows: {}\tBatches: {}".format(rows, batches))

    ids = data.index.to_numpy()
    ids = ids.reshape(ids.shape[0], 1)
    
    # poster = imread(image_url)
    # poster = np.reshape(poster, (1, poster.shape[0] * poster.shape[1] * poster.shape[2]))
    pca = IncrementalPCA(components, batch_size = batch_size)

    # imgs = np.zeros((batch_size, img_dim))
    print("Start PCAing")
    for i in range(0, batches*batch_size, batch_size):
        ii = i % batch_size
        imgs = np.apply_along_axis(load_img, 1, ids[i : i + batch_size])
        # imgs[i] = imread(path + "normal\\{}.jpg".format()).flatten()
        print("Fitting {} of {}".format(int(i / batch_size) + 1, batches))
        pca.partial_fit(imgs)
    
    if(rest > 0): 
        print("Fitting the rest")
        imgs = np.apply_along_axis(load_img, 1, ids[batches*batch_size : batches*batch_size + rest])


    plt.grid()
    plt.plot(pca.explained_variance_ratio_)
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
    # components = 1024
    # path = os.getcwd() + "/data/duplicate_free_41K.csv"
    main()