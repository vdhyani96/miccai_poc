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


img = Image.open('/data02/shared/vikasd/scripts/poc/data/segmentation_training/gbm/training-set/image01.png')
X = TF.to_tensor(img)

print(X.shape)
print(img.size)

# check num channels in image
if X.shape[0] == 4:
    print("Hey, there are 4 channels")

# printing each channel one by one to see which channel to throw away
# we throw the last channel away
X = X[:3, :, :]
print("Modified tensor shape after removing one channel =", X.shape)

# now creating 224x224 overlapping patches

patches = X.unfold(1, 224, 100).unfold(2, 224, 100)
print(patches.shape)

"""
pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
print(pil_to_tensor.shape)

tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)
"""