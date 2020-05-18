# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:11:45 2020

@author: DHYANI
"""
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import numpy as np

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
        patches = X.unfold(0, 3, 3).unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        # ([1, 4, 4, 3, 224, 224]) >> channel, patchcount_x, patchcount_y, patchsize_x, patchsize_y
        #print("Patches list dimension", patches.shape, "and size", torch.prod(torch.tensor(patches.shape)))
        patches = patches.reshape(-1, 3, patch_size, patch_size)
        #print("After resize: Patches list dimension", patches.shape)
        
        # now reading the corresponding segmentation mask of the image and extract patches
        img_name = ID.split('/')[-1]
        img_id = img_name.split('.')[0]
        mask_name = img_id + "_mask.txt"
        mask_path = "/".join(ID.split('/')[:-1]) + "/{}".format(mask_name)
        #print("Mask path is", mask_path)
        
        # now read the mask
        if ID in self.mag20x:
            # double the mask size to 40x
            height = 2*height
            width = 2*width
        mask = np.zeros((height * width))
        j = -1                                                                                                                       
        with open(mask_path) as file:                                                                                                
            for line in file:                                                                                                        
                # print(line)                                                                                                        
                if j >= 0:  # skip first line, first line is image size                                                              
                    # also binarize the mask
                    mask[j] = 1 if int(line) > 0 else 0
                j += 1
        mask = mask.reshape(height, width)
        mask = torch.from_numpy(mask)
        
        #print("Mask: shape is", mask.shape)
        
        # now extract patches out of the mask
        mask_patches = mask.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        #print("Mask: Patches list dimension", mask_patches.shape, "and size", torch.prod(torch.tensor(mask_patches.shape)))
        mask_patches = mask_patches.reshape(-1, 1, patch_size, patch_size)
        #print("Mask: After resize: Patches list dimension", mask_patches.shape)
        
        return [patches, mask_patches], y