from sklearn.decomposition import IncrementalPCA
from skimage.io import imread
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

img_dim = 320 * 256 * 3
path = "data/"
batch_size = 1000
components = None

def load_img(id):
    return imread(path + "normal/{}.jpg".format(id[0])).astype(np.int16).flatten()

def main(pca_path = None):
    pca = IncrementalPCA(components, batch_size = batch_size)
       
    data = pd.read_csv(path + "41K_processed_v3.csv", index_col='id')
    data = data[data["cols"] != 0]
    ids = data.index.to_numpy()
    ids = ids.reshape(ids.shape[0], 1)
    
    fit(pca, ids)
    

def fit(pca: IncrementalPCA, ids):
    rows = len(ids)
    # rows = 1000
    batches = int(np.floor(rows / batch_size))
    rest = rows - batches * batch_size

    print("Rows: {}\tBatches: {}".format(rows, batches))

    print("Start fitting PCA")
    for i in range(0, (batches - 1) * batch_size, batch_size):
        imgs = np.apply_along_axis(load_img, 1, ids[i : i + batch_size])
        # imgs[i] = imread(path + "normal\\{}.jpg".format()).flatten()
        print("Fitting {} of {}".format(int(i / batch_size) + 1, batches))
        pca.partial_fit(imgs)
    
    if(rest > 0): 
        print("Fitting {} of {}".format(int((i + 1) / batch_size) + 1, batches))
        imgs = np.apply_along_axis(load_img, 1, ids[(batches - 1) * batch_size : (batches - 1) * batch_size + rest])
    
    plt.grid()
    plt.title("Explained variance.")
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    
    if(components != None): plt.savefig('Scree_{}_components.png'.format(components))
    else: plt.savefig('Scree_all_components.png')
        
def generate_pca_dataset(pca):
    pass

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