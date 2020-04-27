# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:14:48 2020

@author: DHYANI
"""

from glob import glob
import random

import torch
from torch.utils import data

from my_classes import Dataset


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# Parameters
params = {'batch_size': 3,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 2


# Datasets
partition, labels = {'train': [], 'validation': []}, {}

data_root = '/data02/shared/vikasd/scripts/poc/data'
gbm_train = glob('{}/segmentation_training/gbm/training-set/*.png'.format(data_root))
hnsc_train = glob('{}/segmentation_training/hnsc/training-set/*.png'.format(data_root))
lgg_train = glob('{}/segmentation_training/lgg/training-set/*.png'.format(data_root))
lung_train = glob('{}/segmentation_training/lung/training-set/*.png'.format(data_root))
classes = [gbm_train, hnsc_train, lgg_train, lung_train]

# populate the dictionary partition and label
# split 75-25% between train-validation
for i in range(len(classes)):
    for img_path in classes[i]:
        img_name = img_path.split('/')[-1]
        img_id = img_name.split('.')[0]
        
        # if this is the image file we need
        # 75-25 split using uniform distribution
        if img_id[-1].isdigit():
            if random.uniform(0,1) > 0.25:
                partition['train'].append(img_path)
            else:
                partition['validation'].append(img_path)
            labels[img_path] = i


# Check if the dictionaries were populated correctly
#print(partition)
print("Train and validation data count: ")
print(len(partition['train']))
print(len(partition['validation']))
print("\nNow the labels count:")
print(len(labels))

# Now I have the dictionaries ready, I can go ahead
# and create the generators for training and validation set



# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)



# Loop over epochs
for epoch in range(max_epochs):
    # Training
    i = 0
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        i += 1
        print(i, "batch")
        print("local_batch shape", local_batch.shape)
        print("local_labels shape", local_labels.shape)
        
        # Model computations
        #[...]

"""
    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]
"""