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
        # Select sample, ID is basically the path of image
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
        
        # now handle 4-channel images
        if X.shape[0] == 4:
            X = X[:3, :, :]
        
        
        # now need to extract patches out of the loaded image
        patch_size = 224
        stride = 100
        patches = X.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        # ([3, 4, 4, 224, 224]) >> channel, patchcount_x, patchcount_y, patchsize_x, patchsize_y
        print("Patches list dimension", patches.shape)
        
        return patches, y