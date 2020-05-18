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
params = {'batch_size': 1,
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
            
            del X, y, output
            torch.cuda.empty_cache()
        
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
                
                del Xval, yval, valOutput
                torch.cuda.empty_cache()
        
        print("Loss value =", sumloss / len(local_batch))
        
        avgDice = sumDice / num
        diceList.append(avgDice)
        print("Dice score on validation set =", avgDice)
    print("\nAverage Dice score for the current epoch =", np.mean(avgDice))



# Save the trained UNET model:
unetpath = './model/unet.pth'
torch.save(unet.state_dict(), unetpath)


# free CUDA
#del X, y, Xval, yval, val_batch, local_batch
#torch.cuda.empty_cache()

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

params2 = {'batch_size': 3,
          'shuffle': True,
          'collate_fn': my_collate,
          'num_workers': 6}

# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params2)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params2)

vgg_epochs = 3

# using the vgg16 pretrained model
vgg16 = models.vgg16_bn(pretrained=True).cuda()

# Freeze model weights -- don't freeze because it's pretrained on
# natural images
#for param in vgg16.parameters():
#    param.requires_grad = False

# 1. replace the first convolution layer with another one with 4 filters
out_channels = vgg16.features[0].out_channels
layers = list(vgg16.features.children())[1:] # Remove the first layer
newlayer = nn.Conv2d(4, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
layers.insert(0, newlayer) # add the newlayer to the front
vgg16.features = nn.Sequential(*layers)


# 2. replace the last fully connected layer with another one with 4 output features
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(classes))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
#print(vgg16)
vgg16.cuda()

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss() #nn.NLLLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=0.001)

# The pretrained model is ready here - can start training
# on combined 4-D images

print("\n\n================= Starting to train the classification network =================\n")

batch_size = 8
# start training the classification model
vgg16.train()
# Loop over epochs
for epoch in range(vgg_epochs):
    # Training
    i = 0
    
    # preparing some random data
    data_storage, label_storage = None, []
    for local_batch, local_labels in training_generator:
        
        for j in range(len(local_batch)):
            if data_storage is None:
                data_storage = local_batch[j][0].detach().clone()
                y = [local_labels[j] for i in range(local_batch[j][0].shape[0])]
                label_storage.extend(y)
            else:
                data_storage = torch.cat((data_storage, local_batch[j][0]), 0)
                y = [local_labels[j] for i in range(local_batch[j][0].shape[0])]
                label_storage.extend(y)
            #print("data storage shape =", data_storage.shape)
            #print("label_storage shape =", len(label_storage))
            
        # now randomize them
    num = len(label_storage)
    idx = torch.randperm(num)
    label_storage = torch.tensor(label_storage)
    label_storage = label_storage[idx].detach().clone()
    data_storage = data_storage[idx].detach().clone()
    
    print('finally after randomizing the stuff, the shapes are {} and {}'.format(data_storage.shape, label_storage.shape))
    print("Some values of labels:", label_storage[:5])
    
    iterlist = list(range(0, num, batch_size))
    flag = 0
    sumloss = 0
    for idx in iterlist:
        i += 1
        print("\n")
        print(i, "batch")
        print("local_batch size", batch_size)
        correctus = 0
        total_patch = 0
        if num - idx > batch_size:
            rgb = Variable(data_storage[idx:idx+batch_size]).cuda()
            y = Variable(label_storage[idx:idx+batch_size]).cuda()
        else:
            rgb = Variable(data_storage[idx:]).cuda()
            y = Variable(label_storage[idx:]).cuda()
            flag = 1
        mask = unet(rgb)
        image4d = torch.cat((rgb, mask), 1).cuda()
        optimizer.zero_grad()
        
        # forward+backward+optimize
        outputs = vgg16(image4d)
        loss = criterion(outputs, y)
        sumloss += loss.item()
        print("VGG Training Loss:", loss.item())
        
        loss.backward()
        optimizer.step()
        
        # training accuracy
        _, pred = torch.max(outputs, 1)
        correct_pred = torch.sum(y == pred).item()
        tot = y.shape[0]
        print("Actual Label:", y)
        print("Predicted label:", pred)
        print("Training accuracy:", correct_pred/tot)
        
        del rgb, mask, image4d, outputs
        torch.cuda.empty_cache()
        
        val_data_storage, val_label_storage = None, []
        for local_batch, local_labels in validation_generator:
            
            for j in range(len(local_batch)):
                if val_data_storage is None:
                    val_data_storage = local_batch[j][0].detach().clone()
                    y = [local_labels[j] for i in range(local_batch[j][0].shape[0])]
                    val_label_storage.extend(y)
                else:
                    val_data_storage = torch.cat((val_data_storage, local_batch[j][0]), 0)
                    y = [local_labels[j] for i in range(local_batch[j][0].shape[0])]
                    val_label_storage.extend(y)
                #print("data storage shape =", data_storage.shape)
                #print("label_storage shape =", len(label_storage))
                
        # now randomize them
        val_num = len(val_label_storage)
        val_idx = torch.randperm(val_num)
        val_label_storage = torch.tensor(val_label_storage)
        val_label_storage = val_label_storage[val_idx].detach().clone()
        val_data_storage = val_data_storage[val_idx].detach().clone()
        
        val_batchsize = 10
        val_iterlist = list(range(0, val_num, val_batchsize))
        val_flag = 0
        
        val_correctus = 0
        val_tot = 0
        for validx in val_iterlist:
            
            if val_num - validx > val_batchsize:
                rgb_val = Variable(val_data_storage[validx:validx+val_batchsize]).cuda()
                y = Variable(val_label_storage[validx:validx+val_batchsize]).cuda()
            else:
                rgb_val = Variable(val_data_storage[validx:]).cuda()
                y = Variable(val_label_storage[validx:]).cuda()
                val_flag = 1
            mask = unet(rgb_val)
            image4d_val = torch.cat((rgb_val, mask), 1).cuda()
            valOutput = vgg16(image4d_val)
            _, pred = torch.max(valOutput, 1)
            correct_pred = torch.sum(y == pred).item()
            val_correctus += correct_pred
            tot = y.shape[0]
            val_tot += tot
            #print("Val:Actual Label:", y)
            #print("Val:Predicted label:", pred)
            #val_acc = correct_pred/tot
            #print("Validation accuracy:", val_acc)
            
            
            del rgb_val, mask, image4d_val, valOutput, y, pred
            torch.cuda.empty_cache()
            
            if val_flag == 1:
                break
        
        print("Average Validation accuracy:", val_correctus/val_tot)
        
        if flag == 1:
            break
        
        
            

        

        
"""       
        # prediction
            rgb_val = Variable(val_data_storage).cuda()
            mask = unet(rgb_val)
            image4d_val = torch.cat((rgb_val, mask), 1).cuda()
            valOutput = vgg16(image4d_val)
            y = Variable(val_label_storage)
            
            _, pred = torch.max(valOutput, 1)
            correct_pred = torch.sum(y == pred).item()
            tot = y.shape[0]
            print("Val:Actual Label:", y)
            print("Val:Predicted label:", pred)
            print("Validation accuracy:", correct_pred/tot)
        
        del rgb_val, mask, image4d_val, valOutput, y, pred
        torch.cuda.empty_cache()
        
        if flag == 1:
            break
    
    
    

    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        i += 1
        sumloss = 0
        print("\n")
        print(i, "batch")
        print("local_batch size", len(local_batch))
        correctus = 0
        total_patch = 0
        for j in range(len(local_batch)):
            # change dimension of y
            ylist = [local_labels[j] for i in range(local_batch[j][0].shape[0])]
            y = Variable(torch.tensor(ylist).cuda())
            #print("local element shape for element", j+1, "is", len(local_batch[j]))
            #print("local image shape is", local_batch[j][0].shape)
            #print("local mask shape is", local_batch[j][1].shape)
            rgb = Variable(local_batch[j][0]).cuda()
            mask = unet(rgb)
            # create 4D image by concatenating the RGB image with its
            # segmentation mask
            image4d = torch.cat((rgb, mask), 1).cuda()
            #print("shape of the 4-D image =",image4d.shape)
            print("Printing the 4-D image values. 0 channel:", image4d[0,0,:3,:3])
            print("Printing the 4-D image values. 1 channel:", image4d[0,1,:3,:3])
            print("Printing the 4-D image values. 2 channel:", image4d[0,2,:3,:3])
            print("Printing the 4-D image values. 3 channel:", image4d[0,3,:3,:3])
            
            optimizer.zero_grad()
            
            # forward+backward+optimize
            outputs = vgg16(image4d)
            #print("vgg output shape:", outputs.shape)
            #print(outputs)
            #print(y)
            loss = criterion(outputs, y)
            sumloss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # training accuracy
            total_patch += local_batch[j][0].shape[0]
            _, pred = torch.max(outputs, 1)
            correctus += torch.sum(y == pred).item()
            print("Actual Label:", y)
            print("Predicted label:", pred)
            
            del rgb, mask, image4d, outputs
            torch.cuda.empty_cache()
        
        correctus_val = 0
        total_patch_val = 0
        for val_batch, val_labels in validation_generator:
            for j in range(len(val_batch)):
                rgb_val = Variable(val_batch[j][0]).cuda()
                mask = unet(rgb_val) #.cuda()
                image4d_val = torch.cat((rgb_val, mask), 1).cuda()
                valOutput = vgg16(image4d_val)
                
                ylist = [val_labels[j] for i in range(val_batch[j][0].shape[0])]
                y = torch.tensor(ylist).cuda()
                
                total_patch_val += val_batch[j][0].shape[0]
                _, pred = torch.max(valOutput, 1)
                correctus_val += torch.sum(y == pred).item()
                #print("Validation set ground truth:", y)
                #print("Validation set prediction:", pred)
                
                del rgb_val, mask, image4d_val, valOutput, y, pred
                torch.cuda.empty_cache()
        
        print("Total patches={} and total correct={}".format(total_patch, correctus))
        print("Training Accuracy =", correctus / total_patch)
        print("Val: Total patches={} and total correct={}".format(total_patch_val, correctus_val))
        print("Validation Accuracy =", correctus_val / total_patch_val)
        print("VGG Loss value =", sumloss / len(local_batch))
        




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