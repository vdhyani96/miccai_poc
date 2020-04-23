# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:14:48 2020

@author: DHYANI
"""

from glob import glob
import torch
from torch.utils import data

from my_classes import Dataset


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100


# Datasets
partition, labels = {'train': [], 'validation': []}, {}

data_root = '/data02/shared/vikasd/scripts/poc/data'
gbm_train = glob('{}/segmentation_training/gbm/training-set/*.png'.format(data_root))
hnsc_train = glob('{}/segmentation_training/hnsc/training-set/*.png'.format(data_root))
lgg_train = glob('{}/segmentation_training/lgg/training-set/*.png'.format(data_root))
lung_train = glob('{}/segmentation_training/lung/training-set/*.png'.format(data_root))
classes = [gbm_train, hnsc_train, lgg_train, lung_train]

# populate the dictionary partition and label
for i in range(len(classes)):
    for img_path in classes[i]:
        img_name = img_path.split('/')[-1]
        img_id = img_name.split('.')[0]
        
        # if this is the image file we need
        if img_id[-1].isdigit():
            partition['train'].append(img_path)
            labels[img_path] = i


# Check if the dictionaries were populated correctly
print(partition)
print("\nNow the labels:")
print(labels)


"""
# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        [...]

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]
"""