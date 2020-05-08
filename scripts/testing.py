# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 03:38:45 2020

@author: DHYANI
"""

from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np


#img = Image.open('/data02/shared/vikasd/scripts/poc/data/segmentation_training/gbm/training-set/image06_mask.png')
img = Image.open('/data02/shared/vikasd/scripts/poc/data/segmentation_training/gbm/training-set/image06.png')
mask_path = '/data02/shared/vikasd/scripts/poc/data/segmentation_training/gbm/training-set/image06_mask.txt'
img_W, img_H = img.size
mask = np.zeros((img_H * img_W,))
j = -1                                                                                                                       
with open(mask_path) as file:                                                                                                
    for line in file:                                                                                                        
        # print(line)                                                                                                        
        if j >= 0:  # skip first line, first line is image size                                                              
            mask[j] = int(line)                                                                                              
        j += 1
mask = mask.reshape(img_H, img_W)


#X = TF.to_tensor(img)
X = torch.from_numpy(mask)

print(X.shape)
#print(img.size)

# check num channels in image
if X.shape[0] == 4:
    print("Hey, there are 4 channels")
else:
    print("There are {} channels".format(X.shape[0]))

# printing each channel one by one to see which channel to throw away
# we throw the last channel away
#X = X[:3, :, :]
print("Modified tensor shape after removing one channel =", X.shape)
print("Size is", torch.prod(torch.tensor(X.shape)))

# now creating 224x224 overlapping patches

patches = X.unfold(0, 224, 100).unfold(1, 224, 100)
print(patches.shape)
print("unfolding size is", torch.prod(torch.tensor(patches.shape)))

patches = patches.reshape(-1, 224, 224)
print(patches.shape)



# Test below code to read the segmentation masks properly from txt file
"""
mask_path = '{0}/{1}_mask.txt'.format(folder_path, img_name)                                                                 
mask = np.zeros((img_H * img_W,)).astype(np.uint32)                                                                          
j = -1                                                                                                                       
with open(mask_path) as file:                                                                                                
    for line in file:                                                                                                        
        # print(line)                                                                                                        
        if j >= 0:  # skip first line, first line is image size                                                              
            mask[j] = int(line)                                                                                              
        j += 1                                                                                                               
mask = mask.reshape(img_H, img_W)
"""







"""
pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
print(pil_to_tensor.shape)

tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)
"""