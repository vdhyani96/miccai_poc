# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:11:45 2020

@author: DHYANI
"""
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms.functional as TF

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.mag20x = ['/data02/shared/vikasd/scripts/poc/data/segmentation_training/lung/training-set/image01.png']

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # Get the label
        y = self.labels[ID]
        
        # load the image
        X = Image.open(ID)
        width, height = X.size
        if ID in self.mag20x:
            # double the image size to 40x
            X = X.resize((2*width, 2*height))
        
        # convert to tensor
        X = TF.to_tensor(X)
        
        # now need to extract patches out of the loaded image
        
        return X, y