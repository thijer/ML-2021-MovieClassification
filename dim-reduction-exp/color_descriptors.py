import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import time

# Enable accelleration, needs numba package
use_acceleration = False

path = "data/normal/"
density = 16
maxval = density ** 3
step = 256 / density
# start = int(step / 2)

# Which classifiers to use
c = {
    "RDF": True,
    "DCT": False,
    "KNN": False,
    "MLP": False
}

class_labels = ["action", "adventure", "animation", "comedy", "crime", "drama", "fantasy", "horror", "mystery", "romance", "sci-fi", "short", "thriller"]

# Ridiculous speed improvement using parallelization.
# Needs numba package.
if(use_acceleration):
    from numba import jit
    @jit
    def color2desc(img):
        """ 
        Convert RGB colors to a single color label. 
        """
        desc = np.zeros(img.shape[0], dtype=np.int32)
        for i in range(img.shape[0]):
            ind0 = min(int(img[i, 0] / step), density)
            ind1 = min(int(img[i, 1] / step), density) * density
            ind2 = min(int(img[i, 2] / step), density) * density * density
            desc[i] = ind0 + ind1 + ind2
        return desc

    def img2desc(id):
        img = imread(path + "{}.jpg".format(id[0]))
        img = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 3))
        
        pixel_desc = color2desc(img)
        pixel_desc = np.bincount(pixel_desc, minlength=maxval)
        return pixel_desc

else:
    # Default descriptor extraction without parallelization
    # Let it run overnight.
    def color2desc(rgb: np.ndarray):
        ind0 = int(rgb[0] / step)
        ind1 = int(rgb[1] / step) * density
        ind2 = int(rgb[2] / step) * density * density
        return ind0 + ind1 + ind2

    def img2desc(id):
        img = imread(path + "{}.jpg".format(id[0]))
        img = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 3))
        
        pixel_desc = np.apply_along_axis(color2desc, 1, img)
        pixel_desc = np.bincount(pixel_desc, minlength=maxval)
        return pixel_desc

def main():
    # data = pd.read_csv("data/41K_processed_v3.csv", index_col='id', nrows = 1000)
    data = pd.read_csv("data/41K_processed_v3.csv", index_col='id')
    
    data = data[data["cols"] > 0]
    ids = data.index.to_numpy()
    ids = ids.reshape(ids.shape[0], 1)
    
    Y = data[class_labels].to_numpy()
    # uni, cnt = np.unique(Y, return_counts=True, axis = 0)
    # print(uni)
    # print(cnt)
    
    # Extract color descriptors
    print("Starting extracting color descriptors.")
    start = time.time()
    X = np.apply_along_axis(img2desc, 1, ids)
    print("Color descriptors extraction finished in {:.0f}s".format(time.time() - start))
    
    # Split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    print("Running classifiers")

    if(c["RDF"]):
        from sklearn.ensemble import RandomForestClassifier
        clf_rft = RandomForestClassifier()
        start = time.time()
        clf_rft.fit(X_train, Y_train)
        print("RFT finished in {:.4f}s".format(time.time() - start))
        Y_pred = clf_rft.predict(X_test)
        f1, accuracy, hamming = f1_score(Y_test, Y_pred, average='micro'), accuracy_score(Y_test, Y_pred), hamming_loss(Y_pred, Y_test)
        print("F1: {:.4f}\tAccuracy: {:.4f}\t Hamming loss: {:.4f}".format(f1, accuracy, hamming))
    
    elif(c["DCT"]):
        from sklearn.tree import DecisionTreeClassifier
        clf_dct = DecisionTreeClassifier()
        start = time.time()
        clf_dct.fit(X_train, Y_train)
        print("DCT finished in {:.4f}s".format(time.time() - start))
        Y_pred = clf_dct.predict(X_test)
        f1, accuracy, hamming = f1_score(Y_test, Y_pred, average='micro'), accuracy_score(Y_test, Y_pred), hamming_loss(Y_pred, Y_test)
        print("F1: {:.4f}\tAccuracy: {:.4f}\t Hamming loss: {:.4f}".format(f1, accuracy, hamming))
    
    elif(c["KNN"]):
        from sklearn.neighbors import KNeighborsClassifier
        clf_knn = KNeighborsClassifier(n_neighbors=10)
        start = time.time()
        clf_knn.fit(X_train, Y_train)
        print("KNN finished in {:.4f}s".format(time.time() - start))
        Y_pred = clf_knn.predict(X_test)
        f1, accuracy, hamming = f1_score(Y_test, Y_pred, average='micro'), accuracy_score(Y_test, Y_pred), hamming_loss(Y_pred, Y_test)
        print("F1: {:.4f}\tAccuracy: {:.4f}\t Hamming loss: {:.4f}".format(f1, accuracy, hamming))

    elif(c["MLP"]):
        from sklearn.neural_network import MLPClassifier
        clf_mlp = MLPClassifier(activation='tanh')
        # clf_mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(300, 350))
        start = time.time()
        clf_mlp.fit(X_train, Y_train)
        print("MLP finished in {:.4f}s".format(time.time() - start))
        Y_pred = clf_mlp.predict(X_test)
        f1, accuracy, hamming = f1_score(Y_test, Y_pred, average='micro'), accuracy_score(Y_test, Y_pred), hamming_loss(Y_pred, Y_test)
        print("F1: {:.4f}\tAccuracy: {:.4f}\t Hamming loss: {:.4f}".format(f1, accuracy, hamming))
    
    print("It is Done.")

if __name__ == '__main__':
    main()