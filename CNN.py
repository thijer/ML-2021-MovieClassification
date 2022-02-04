import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=4, padding=(0,1))
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=(0,1))
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=(0,1))

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.pool_last = nn.MaxPool2d(2, 2, padding=(0,1), ceil_mode=True)

        self.fc1 = nn.Linear(768, 768)
        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 13)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool_last(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
