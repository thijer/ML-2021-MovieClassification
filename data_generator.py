import torch
import cv2
from PIL import Image
import numpy
class data_generator(torch.utils.data.Dataset):
  def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = numpy.asarray(Image.open('../normal/' + str(ID) + '.jpg'), dtype=numpy.float32) / 255

        X = X.transpose(2, 0, 1)
        y = self.labels[ID]

        return X, y
