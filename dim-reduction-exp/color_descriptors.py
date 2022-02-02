import matplotlib as mpl
import numpy as np
from skimage.io import imread
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle, resample
import time
from numba import jit

path = "data/normal/"
density = 10
maxval = density ** 3
step = 256 / density
# start = int(step / 2)

class_labels = ["action", "adventure", "animation", "comedy", "crime", "drama", "fantasy", "horror", "mystery", "romance", "sci-fi", "short", "thriller"]

# def color2desc(rgb):
#     return int(rgb / step)

# Ridiculous speed improvement using parallelization
@jit
def color2desc_prl(img):
    """ 
    Convert RGB colors to a single color label. """
    desc = np.zeros(img.shape[0], dtype=np.int32)
    for i in range(img.shape[0]):
        ind0 = min(int(img[i, 0] / step), density)
        ind1 = min(int(img[i, 1] / step), density) * density
        ind2 = min(int(img[i, 2] / step), density) * density * density
        desc[i] = ind0 + ind1 + ind2
    return desc

def img2desc_prl(id):

    img = imread(path + "{}.jpg".format(id[0]))
    img = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 3))
    # imgsize = img.shape[0]
    # desc = np.vectorize(color2desc)
    # pixel_desc = desc(img)
    pixel_desc = color2desc_prl(img)
    pixel_desc = np.bincount(pixel_desc, minlength=maxval)
    # print(imgsize, np.sum(pixel_desc))
    return pixel_desc

# Default descriptor extraction without parallelization
def color2desc(rgb: np.ndarray):
    ind0 = int(rgb[0] / step) * 1
    ind1 = int(rgb[1] / step) * 8 * 1
    ind2 = int(rgb[2] / step) * 8 * 8 * 1
    return ind0 + ind1 + ind2

def img2desc(id):

    img = imread(path + "{}.jpg".format(id[0]))
    img = np.reshape(img, newshape=(img.shape[0] * img.shape[1], 3))
    # imgsize = img.shape[0]
    # desc = np.vectorize(color2desc)
    # pixel_desc = desc(img)
    pixel_desc = np.apply_along_axis(color2desc, 1, img)
    pixel_desc = np.bincount(pixel_desc, minlength=maxval)
    # print(imgsize, np.sum(pixel_desc))
    return pixel_desc

def test():
    img = imread(path + "3.jpg")
    img2desc(img)
    pass


def main():
    # data = pd.read_csv("data/41K_processed_v3.csv", index_col='id', nrows = 1000)
    data = pd.read_csv("data/41K_processed_v3.csv", index_col='id')
    print(len(data))
    data = data[data["cols"] > 0]
    
    print(len(data))
    ids = data.index.to_numpy()
    ids = ids.reshape(ids.shape[0], 1)
    # uni, cnt = np.unique(Y, return_counts=True, axis = 0)
    # print(uni)
    # print(cnt)
    
    Y = data[class_labels].to_numpy()

    # Y = data["genre"].to_numpy()
    # enc = LabelEncoder()
    # Y = enc.fit_transform(Y)

    # Extract color descriptors
    start = time.time()
    X = np.apply_along_axis(img2desc_prl, 1, ids)
    print("Color descriptors extraction finished in {}s".format(time.time() - start))
    
    # start = time.time()
    # X = np.apply_along_axis(img2desc, 1, ids)
    # print("{} color descriptors extracted in {}s".format(X.shape[0], time.time() - start))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    # clf_dct = DecisionTreeClassifier()
    # clf_knn = KNeighborsClassifier(n_neighbors=10)
    # clf_mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=500)
    # clf_gnb = GaussianNB()
    # clf_svm = SVC()
    clf_rft = RandomForestClassifier()
    
    # clf_vot = VotingClassifier(
    #     [("DCT", clf_dct), ("KNN", clf_knn), ("MLP", clf_mlp), ("GNB", clf_gnb), ("SVM", clf_svm)],
    #     n_jobs=-1
    # )
    
    print("Starting fitting")
    
    # start = time.time()
    # clf_vot.fit(X_train, Y_train)
    # print("Fitting finished in {:d}s".format(time.time() - start))
    # Y_pred = clf_vot.predict(X_test)
    # f1, accuracy = f1_score(Y_pred, Y_test, average='micro'), accuracy_score(Y_pred, Y_test)
    # print("F1: {:.4f}\tAccuracy: {:.4f}".format(f1, accuracy))
    
    start = time.time()
    clf_rft.fit(X_train, Y_train)
    print("RFT finished in {}s".format(time.time() - start))
    Y_pred = clf_rft.predict(X_test)
    f1, accuracy = f1_score(Y_pred, Y_test, average='micro'), accuracy_score(Y_pred, Y_test)
    print("F1: {:.4f}\tAccuracy: {:.4f}".format(f1, accuracy))
    
    
    # start = time.time()
    # clf_dct.fit(X_train, Y_train)
    # print("DCT finished in {}s".format(time.time() - start))
    # Y_pred = clf_dct.predict(X_test)
    # f1, accuracy = f1_score(Y_pred, Y_test, average='micro'), accuracy_score(Y_pred, Y_test)
    # print("F1: {:.4f}\tAccuracy: {:.4f}".format(f1, accuracy))
    
    # start = time.time()
    # clf_knn.fit(X_train, Y_train)
    # print("KNN finished in {}s".format(time.time() - start))
    # Y_pred = clf_knn.predict(X_test)
    # f1, accuracy = f1_score(Y_pred, Y_test, average='micro'), accuracy_score(Y_pred, Y_test)
    # print("F1: {:.4f}\tAccuracy: {:.4f}".format(f1, accuracy))
    
    # start = time.time()
    # clf_mlp.fit(X_train, Y_train)
    # print("MLP finished in {}s".format(time.time() - start))
    # Y_pred = clf_mlp.predict(X_test)
    # f1, accuracy = f1_score(Y_pred, Y_test, average='micro'), accuracy_score(Y_pred, Y_test)
    # print("F1: {:.4f}\tAccuracy: {:.4f}".format(f1, accuracy))
    
    
    

if __name__ == '__main__':
    main()