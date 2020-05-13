# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:14:48 2020

@author: DHYANI
"""

from glob import glob
import random
import numpy as np

import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import models

from my_classes import Dataset
from unet import Unet



# function to calculate the dice metric for segmentation performance
def dice_coef(output, target):
    smooth = 1.

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return 1 - ((2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth))


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


# Function to provide the training and validation data
# as well as the labels in form of dictionary
def dataProvider(classes):
    # populate the dictionary partition and label
    # split 75-25% between train-validation
    partition, labels = {'train': [], 'validation': []}, {}
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
    return partition, labels


output_path = '/data02/shared/vikasd/scripts/poc/output/unet'
data_root = '/data02/shared/vikasd/scripts/poc/data'
gbm_train = glob('{}/segmentation_training/gbm/training-set/*.png'.format(data_root))
hnsc_train = glob('{}/segmentation_training/hnsc/training-set/*.png'.format(data_root))
lgg_train = glob('{}/segmentation_training/lgg/training-set/*.png'.format(data_root))
lung_train = glob('{}/segmentation_training/lung/training-set/*.png'.format(data_root))
classes = [gbm_train, hnsc_train, lgg_train, lung_train]

# Datasets
partition, labels = dataProvider(classes)

# Check if the dictionaries were populated correctly
#print(partition)
print("Train and validation data count: ")
print(len(partition['train']))
print(len(partition['validation']))
print("\nNow the labels count:")
print(len(labels))



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

# Parameters
params = {'batch_size': 3,
          'shuffle': True,
          'collate_fn': my_collate,
          'num_workers': 6}
max_epochs = 1

# Since I have the dictionaries ready, I can go ahead
# and create the generators for training and validation set

# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)


# creating the UNET segmentation model object 
unet = Unet(input_nc = 3, output_nc = 1, num_downs = 5, output_actfun = nn.Sigmoid()).cuda()
optim = torch.optim.Adam(unet.parameters(), lr=0.001)
bceloss = torch.nn.BCEWithLogitsLoss()


# start training the segmentation model
unet.train()
# Loop over epochs
for epoch in range(max_epochs):
    # Training
    i = 0
    diceList = []
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        i += 1
        sumloss = 0
        print("\n")
        print(i, "batch")
        print("local_batch size", len(local_batch))
        for j in range(len(local_batch)):
            #print("local element shape for element", j+1, "is", len(local_batch[j]))
            print("local image shape is", local_batch[j][0].shape)
            #print("local mask shape is", local_batch[j][1].shape)
            X = Variable(local_batch[j][0]).cuda()
            y = Variable(local_batch[j][1]).cuda()
            output = unet(X)
            #print("y shape", y.shape)
            print("predicted y shape", output.shape)
            # binarize the output using 0.5 as threshold
            #output[output < 0.5] = 0
            #output[output >= 0.5] = 1
            loss = bceloss(output, y)
            sumloss += loss.item()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        sumDice = num = 0
        for val_batch, val_labels in validation_generator:
            for j in range(len(val_batch)):
                Xval = Variable(val_batch[j][0]).cuda()
                yval = Variable(val_batch[j][1]).cuda()
                valOutput = unet(Xval)
                # binarize the output
                valOutput[valOutput < 0.5] = 0
                valOutput[valOutput >= 0.5] = 1
                diceMetric = dice_loss(valOutput, yval)
                #print("dice value for one val image =", diceMetric)
                #print("yval", torch.unique(yval))
                #print("valOutput", torch.unique(valOutput))
                sumDice += diceMetric.item()
                num += 1
                
                # save one of the patches along with corresponding
                # ground truth and the actual RGB patch
                save_image(Xval[3, :, :, :], '{}/rgbimg_{}.png'.format(output_path, num))
                save_image(yval[3, :, :, :], '{}/groundtruth_{}.png'.format(output_path, num))
                save_image(valOutput[3, :, :, :], '{}/predicted_{}.png'.format(output_path, num))
        
        print("Loss value =", sumloss / len(local_batch))
        
        avgDice = sumDice / num
        diceList.append(avgDice)
        print("Dice score on validation set =", avgDice)
    print("\nAverage Dice score for the current epoch =", np.mean(avgDice))



##---------- Classification Network below -----------------


# Generating the data again with random train-val set
partition, labels = dataProvider(classes)

# Check if the dictionaries were populated correctly
#print(partition)
print("VGG16: Train and validation data count: ")
print(len(partition['train']))
print(len(partition['validation']))
print("\nVGG16: Now the labels count:")
print(len(labels))


# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)

vgg_epochs = 2

# using the vgg16 pretrained model
vgg16 = models.vgg16_bn(pretrained=True)

# Freeze model weights
for param in vgg16.parameters():
    param.requires_grad = False

# replace the last fully connected layer with another one with 4 output features
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(classes))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
#print(vgg16)

# The pretrained model is ready here - can start training
# on combined 4-D images

print("\n\n================= Starting to train the classification network =================\n")
# start training the classification model
vgg16.train()
# Loop over epochs
for epoch in range(vgg_epochs):
    # Training
    i = 0
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        i += 1
        sumloss = 0
        print("\n")
        print(i, "batch")
        print("local_batch size", len(local_batch))
        for j in range(len(local_batch)):
            #print("local element shape for element", j+1, "is", len(local_batch[j]))
            print("local image shape is", local_batch[j][0].shape)
            #print("local mask shape is", local_batch[j][1].shape)
            rgb = Variable(local_batch[j][0]).cuda()
            mask = unet(rgb)
            # create 4D image by concatenating the RGB image with its
            # segmentation mask
            image4d = torch.cat((rgb, mask), 1)
            print("shape of the 4-D image =",image4d.shape)
            
            #loss = bceloss(output, y)
            #sumloss += loss.item()
            
            #optim.zero_grad()
            #loss.backward()
            #optim.step()







"""
# Loop over epochs
for epoch in range(max_epochs):
    # Training
    i = 0
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        i += 1
        print("\n")
        print(i, "batch")
        print("local_batch shape", len(local_batch))
        for j in range(len(local_batch)):
            print("local element shape for element", j+1, "is", len(local_batch[j]))
            print("local image shape is", local_batch[j][0].shape)
            print("local mask shape is", local_batch[j][1].shape)
        print("local label", local_labels, "and local_labels shape", len(local_labels))
        print("\n\n")
        # Model computations
        #[...]


    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]
"""