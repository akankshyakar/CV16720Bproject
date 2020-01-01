#!/usr/bin/env python
# coding: utf-8

# In[92]:
import os, os.path

import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as utils
import torch.optim as optim      # implementing various optimization algorithms
import matplotlib.pyplot as plt
import numpy as np

import gzip
import _pickle as cPickle
import os
from collections import Counter
# device = torch.device("cuda" if use_cuda else "cpu")

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KDTree

np.random.seed(123)
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")

with open('../txtfile/train_classes.txt', 'r') as infile:
    train_classes = [str.strip(line) for line in infile]
# print(train_classes)
with open('../txtfile/zsl_classes.txt', 'r') as infile:
    zsl_classes = [str.strip(line) for line in infile]

DATAPATH= "../data/zeroshot_data.pkl"
# criterion=nn.CrossEntropyLoss()
criterion=nn.NLLLoss()

NUM_CLASS = 15
NUM_ATTR = 300
BATCH_SIZE = 128
EPOCH = 10


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
    
def load_data():
    """read data, create datasets"""
    # READ DATA
    with gzip.GzipFile(DATAPATH, 'rb') as infile:
        data = cPickle.load(infile)

    # ONE-HOT-ENCODE DATA
    label_encoder   = LabelEncoder()
    label_encoder.fit(train_classes)

    training_data = [instance for instance in data if instance[0] in train_classes]
    zero_shot_data = [instance for instance in data if instance[0] not in train_classes]
    # SHUFFLE TRAINING DATA
    np.random.shuffle(training_data)

    ### SPLIT DATA FOR TRAINING
    train_size  = 300
    train_data  = list()
    valid_data  = list()
    for class_label in train_classes:
        ct = 0
        for instance in training_data:
            if instance[0] == class_label:
                if ct < train_size:
                    train_data.append(instance)
                    ct+=1
                    continue
                valid_data.append(instance)

    # SHUFFLE TRAINING AND VALIDATION DATA
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    train_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15))for instance in train_data]
    valid_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15)) for instance in valid_data]
    # print("traind atashape",train_data[0][1])
    # FORM X_TRAIN AND Y_TRAIN
    x_train, y_train    = zip(*train_data)
    x_train, y_train    = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))
    # L2 NORMALIZE X_TRAIN
    x_train = normalize(x_train, norm='l2')

    # FORM X_VALID AND Y_VALID
    x_valid, y_valid = zip(*valid_data)
    x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))
    # L2 NORMALIZE X_VALID
    x_valid = normalize(x_valid, norm='l2')


    # FORM X_ZSL AND Y_ZSL
    y_zsl, x_zsl = zip(*zero_shot_data)
    x_zsl, y_zsl = np.squeeze(np.asarray(x_zsl)), np.squeeze(np.asarray(y_zsl))
    # L2 NORMALIZE X_ZSL
    x_zsl = normalize(x_zsl, norm='l2')

    
    tensor_x = torch.stack([torch.Tensor(i) for i in x_train]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in y_train])
    
    my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
    trainset_loader = DataLoader(my_dataset, batch_size=40, shuffle=True, num_workers=0)

    tensor_x = torch.stack([torch.Tensor(i) for i in x_valid]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in y_valid])

    my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
    validset_loader = DataLoader(my_dataset, batch_size=40, shuffle=True, num_workers=0)
    print("-> data loading is completed.")
    return trainset_loader, validset_loader, x_zsl, y_zsl
#     return (x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl)

def get_features(image_name):
    loader = transforms.Compose([ transforms.ToTensor()])
    # print(image_name)
    img = Image.open(image_name).resize((224, 224))
    # plt.imshow(img)
    # plt.show()
    t_img=loader(img).float().unsqueeze(0)
    # t_img = Variable(normalize(ToTensor(scaler(img))).unsqueeze(0))
    # print(t_img.shape)
    vgg16 = models.vgg16(pretrained='imagenet')
    
    # print (vgg16)
    features = list(vgg16.classifier.children())[:1]
    vgg16.classifier = nn.Sequential(*features)
    # print(vgg16)
    # assert(2==3)
    # print(vgg16.classifier[-7])
    # out1=vgg16.features(t_img)
    # print(out1.shape)
    output = vgg16(t_img).detach().numpy()
    # print(output.shape)
    return output

def save_checkpoint(checkpoint_path, model, optimizer):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
    
def valid(model,validset_loader,device,validlis,losslis_val,reshape=False):
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validset_loader:
            if reshape==False:
                target=target.argmax(1).type(torch.long)
            data, target = data.to(device), target.to(device)
            data=data.float()
            target=target.type(torch.long)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validset_loader.dataset)
    losslis_val.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validset_loader.dataset),
        100. * correct / len(validset_loader.dataset)))   
    validlis.append(100. * correct / len(validset_loader.dataset))

def train_save(model,epoch, save_interval, log_interval,trainset_loader,validset_loader,device,optimizer,losslis,acclis,validlis,losslis_val,reshape=False):
    # print(model)
    layers = [x for x in model.parameters()]
    # for l in layers:
    #     print(l.shape,l.requires_grad)
    print("in train_save")
    model.train()  # set training mode
    iteration = 0

    for ep in range(epoch):
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            if reshape==False:
                target=target.argmax(1).type(torch.long)
            data, target = data.to(device), target.to(device)
            data=data.float()
            target=target.type(torch.long)
            # print("datshape",data.shape)
            # print("targetshape",target.shape)
            optimizer.zero_grad()
            output = model(data)
            # print(output.shape)
            # print(output.shape,target.shape)
             
            loss = criterion(output, target)
            # print("herE",torch.sum(model.fc4.weight.data))
            train_loss+=loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
#             acc=np.sum(np.equal(output,target))/target.shape[0]
            
#             total_acc+=acc
            
            loss.backward()
            optimizer.step()
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            # different from before: saving model checkpoints
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('model/zsl_-%i.pth' % iteration, model, optimizer)
            iteration += 1
        train_loss /= len(trainset_loader.dataset)*40
#         total_acc/=len(batches)

        losslis.append(train_loss)
        acclis.append(100.* correct /len(trainset_loader.dataset))
        valid(model,validset_loader,device,validlis,losslis_val,reshape)
    
    # save the final model
    save_checkpoint('model/zsl_-%i.pth' % iteration, model, optimizer)
