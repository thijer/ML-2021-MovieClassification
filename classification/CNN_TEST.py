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
import random
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

if __name__ == '__main__':
    torch.set_printoptions(precision=3)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    params = {'batch_size': 36}

    '''
    path_csv = "data/41K_processed_v2.csv"

    df = pd.read_csv(path_csv, delimiter=',')

    #df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[1:7], axis=1)
    df = df.drop(df.columns[-4:-1], axis=1)

    str_labels=['action', 'adventure', 'animation', 'comedy', 'crime', 'drama', 'fantasy', 'horror', 'mystery', 'romance', 'sci-fi', 'short', 'thriller']
    # User list comprehension to create a list of lists from Dataframe rows
    csv_reader_list = [list(row) for row in df.values]
    images = os.listdir("data/normal")
    ID = np.empty((len(images)), dtype=int)
    labels = {}
    idx = 0

    #print(csv_reader_list)
    for name in images:

        row= df.loc[df['id'] == int(os.path.splitext(name)[0])].values[0]
        if str(row[0]) + '.jpg' in images:
            img = numpy.asarray(Image.open('data/normal/' + str(row[0]) + '.jpg'), dtype=numpy.float32)
            if(len(img.shape)<3):
                continue
            ID[idx] = row[0]
            labels[row[0]] = np.array(row[1:-1])
            idx += 1

    #created training and testing dictionary for the data
    #    np.random.shuffle(labels)
    #print(labels)

    #numpy.save('data/labels.npy', labels, allow_pickle=True)
    #numpy.save('data/ID.npy', ID, allow_pickle=True)
    '''

    labels = numpy.load('data/labels.npy', allow_pickle=True)
    ID = numpy.load('data/ID.npy', allow_pickle=True)

    nsplits = 5
    accuracies = np.zeros(nsplits)
    kf = KFold(n_splits=nsplits, shuffle=True)
    kf.get_n_splits(ID)

    cv_count = 1

    for train_index, test_index in kf.split(ID):

        if not cv_count == 1:
            labels = numpy.load('data/labels.npy', allow_pickle=True)
            ID = numpy.load('data/ID.npy', allow_pickle=True)

        labels = labels.item()

        data = {}
        data['training'] = ID[train_index]
        data['testing'] = ID[test_index]

        training_set = data_generator(data['training'], labels)
        training_generator = torch.utils.data.DataLoader(training_set, shuffle=True, **params)

        net = CNN()
        net.cuda()

        learning_rate = 0.0005
        momentum = 0.9
        weight_decay = 0.0005

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # 0.2827806711055442

        print("INFO: this is cross-validation", cv_count, ". The lr = ", learning_rate,
              ", the momentum = ", momentum, ", the weight_decay = ", weight_decay)
        cv_count += 1

        running_loss = 0

        for epoch in range(1):
            print("Current epoch = ", epoch)
            # Training
            for local_batch, local_labels in training_generator:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                optimizer.zero_grad()
                outputs = net(local_batch)
                loss = criterion(outputs, local_labels.float())
                loss.backward()
                optimizer.step()

        #torch.save(net, "cnn_trained_10epoch_sigmoid_lr_low.pth")

        net = torch.load("models/cnn_trained_10epoch_sigmoid_lr_low.pth")
        net.eval()

        acc=0
        amount = 0

        str_labels=['action', 'adventure', 'animation', 'comedy', 'crime', 'drama', 'fantasy', 'horror', 'mystery', 'romance', 'sci-fi', 'short', 'thriller']
        with torch.no_grad():
            for i in data['testing'][0:]:
                image = numpy.asarray(Image.open('data/normal/' + str(i) + '.jpg'), dtype=numpy.float32)
                x = image.transpose(2, 0, 1)
                x = np.expand_dims(x, axis=0)
                x = torch.from_numpy(x).to(device)
                out_data = net(x)
                highest_idx = torch.argmax(np.round(out_data.cpu(), 3)).item()
                amount += 1
                if labels[i][highest_idx] == 1:
                    acc += 1

                img = cv2.imread('data/normal/' + str(i) + '.jpg')
                img = cv2.putText(img,str_labels[torch.argmax(out_data).item()], (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,2)

                #(this is necessary to avoid Python kernel form crashing)
                cv2.waitKey(0)

                #closing all open windows
                cv2.destroyAllWindows()

        print("The accuracy obtained = ", acc/amount)
        accuracies[cv_count-2] = acc/amount

        torch.save(net, "models/cnn_trained_10epoch_sigmoid_lr_low.pth")
    print(np.sum(np.asarray(accuracies)))
    print(np.sum(accuracies)/5)
