import numpy as np
import os
import csv
import cv2
from data_generator import *
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from CNN import *

import pandas as pd
# Create a dataframe from csv


if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Parameters
    params = {'batch_size': 1}
    #import labels
    path_csv = "data/41K_processed_v2.csv"

    df = pd.read_csv(path_csv, delimiter=',')

    df = df.drop(df.columns[0:7], axis=1)
    df = df.drop(df.columns[-3:], axis=1)

    # User list comprehension to create a list of lists from Dataframe rows
    csv_reader_list = [list(row) for row in df.values]
    images = os.listdir("../normal")
    ID = np.empty((len(images)), dtype=int)
    labels = {}
    idx = 0
    #print(csv_reader_list)
    for name in images:
        img = numpy.asarray(Image.open('../normal/' + name), dtype=numpy.float32)
        if(len(img.shape)<3):
            print(name)
            continue
        ID[idx] = os.path.splitext(name)[0]
        labels[idx] = np.array(csv_reader_list[0])
        idx += 1

    #created training and testing dictionary for the data
    np.random.shuffle(labels)
    data = {}
    data['training'] = ID[0:int(len(labels)*0.8)]
    data['testing'] = ID[int(len(labels)*0.8):]


    training_set = data_generator(data['training'], labels)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    net = CNN()
    net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0
    # Loop over epochs
    for epoch in range(1):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            #print(len(local_batch))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(local_batch)
            loss = criterion(outputs, local_labels.float())
            loss.backward()
            optimizer.step()
