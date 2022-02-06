import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_generator import *
from timeit import default_timer as timer

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = models.vgg16(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

n_inputs = model_ft.classifier[6].in_features
params = {'batch_size': 36}

model_ft.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 768), nn.ReLU(), nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 13))

print(model_ft)

total_params = sum(p.numel() for p in model_ft.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model_ft.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

model_ft = model_ft.to('cuda')

labels = np.load('data/labels.npy', allow_pickle=True)
ID = np.load('data/ID.npy', allow_pickle=True)

labels = labels.item()

data = {}
data['training'] = ID[0:int(len(labels)*0.8)]
data['testing'] = ID[int(len(labels)*0.8):]

training_set = data_generator(data['training'], labels)
training_generator = torch.utils.data.DataLoader(training_set, shuffle=True, **params)

testing_set = data_generator(data['testing'], labels)
testing_generator = torch.utils.data.DataLoader(testing_set, shuffle=True, **params)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)

def train(model,criterion,optimizer,train_loader,valid_loader,n_epochs=5):

    model.epochs = 0
    overall_start = timer()

    for epoch in range(n_epochs):
        print(epoch)
        train_loss = 0.0

        model.train()
        start = timer()

        # Training
        for ii, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            print((ii + 1) / len(train_loader) * 100, train_loss / (ii + 1))
    return model

model = train(
    model_ft,
    criterion,
    optimizer,
    training_generator,
    testing_generator,
    n_epochs=5)

torch.save(model, "models/VGG16_5epoch.pth")
