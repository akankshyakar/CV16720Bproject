#!/usr/bin/env python
# coding: utf-8

# In[92]:

from util import *
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as utils
import torch.optim as optim      # implementing various optimization algorithms

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




WORD2VECPATH    = "../data/class_vectors.npy"
DATAPATH        = "../data/zeroshot_data.pkl"
# criterion=nn.CrossEntropyLoss()
criterion=nn.NLLLoss()
NUM_CLASS = 15
NUM_ATTR = 300
BATCH_SIZE = 128
EPOCH = 65
with open('../txtfile/train_classes.txt', 'r') as infile:
    train_classes = [str.strip(line) for line in infile]
# print(train_classes)
with open('../txtfile/zsl_classes.txt', 'r') as infile:
    zsl_classes = [str.strip(line) for line in infile]


# In[93]:


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(4096, 800)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(800, 300)
        # self.relu2 = nn.ReLU()
        # self.drop2 = nn.Dropout2d(0.5)

        # self.fc3 = nn.Linear(500, 300)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(300,15)
        self.softmax=nn.LogSoftmax()
        
    def forward(self, x):
        # print("in forward")
        # print(x.shape)
        x = self.relu1(self.fc1(x))
#         print(x.shape)
        # x=self.drop1(x)
#         print(x.shape)
        x=self.relu3(self.fc2(x))
#         print(x.shape)
        # x=self.drop2(x)
#         print(x.shape)
        # x = self.relu3(self.fc3(x))
        # print(x.shape)
        # print(self.fc4(x).shape)
        x=self.fc4(x)
        # print("here",x.shape)

        # print(x[0])
        x = self.softmax(x)
        
        return x
def custom_kernel_init(shape,dtype=None):
    class_vectors= np.load(WORD2VECPATH,allow_pickle=True)
    training_vectors= sorted([(label, vec) for (label, vec) in class_vectors if label in train_classes], key=lambda x: x[0])
    classnames, vectors= zip(*training_vectors)
    vectors = np.asarray(vectors, dtype=np.float)
    # print(vectors.shape)
    return vectors

def train_mynet(trainset_loader,testset_loader):
    maxepoch=80
    losslis=[]
    acclis=[]
    validlis=[]
    losslis_val=[]
    epochs=np.arange(maxepoch)
    save_interval, log_interval=500,100

    mymodel = MyNet().to(device)
    numpy_data=custom_kernel_init((300,15),np.float64)
    # print("numpoy shape",numpy_data.shape)
    # print(numpy_data)
    data_tensor=torch.FloatTensor(numpy_data)
    mymodel.fc4.weight.data=data_tensor
    # print(mymodel.fc4.weight.data)
    mymodel.fc4.weight.requires_grad = False
    mymodel.softmax.requires_grad = False

    optimizer = optim.Adam(mymodel.parameters(), lr=5e-5)
    train_save(mymodel,maxepoch, save_interval, log_interval,trainset_loader,testset_loader,device,optimizer,losslis,acclis,validlis,losslis_val,False)


if __name__ == "__main__":

    trainset_loader, validset_loader,x_zsl,y_zsl=load_data()
    train_mynet(trainset_loader,validset_loader)
    #zsl model
    mymodel = MyNet().to(device)
    optimizer = optim.Adam(mymodel.parameters(), lr=5e-5)
    checkpoint=load_checkpoint('model/zsl_-6780_okay.pth', mymodel, optimizer)
    modulelist = list(mymodel.modules())
    model = nn.Sequential(*modulelist[1:-3])
    print(model)
    x_zsl=torch.tensor(x_zsl).float()
    model.eval() 
    pred_zsl = model(x_zsl)
    
    # print(pred_zsl[0][0])
    # print(pred_zsl[1][0])
    # print(pred_zsl[0][1])
    # print(pred_zsl[1][1])

    class_vectors= sorted(np.load(WORD2VECPATH,allow_pickle=True), key=lambda x: x[0])
    classnames, vectors = zip(*class_vectors)
    classnames= list(classnames)
    vectors = np.asarray(vectors, dtype=np.float)

    tree= KDTree(vectors)

    top5, top3, top1 = 0, 0, 0
    for i, pred in enumerate(pred_zsl):
        pred=pred.detach().numpy()
        pred= np.expand_dims(pred, axis=0)
        dist_5, index_5= tree.query(pred, k=5)
        pred_labels= [classnames[index] for index in index_5[0]]
        true_label= y_zsl[i]
        # print(pred_labels,true_label)
        if true_label in pred_labels:
            top5 += 1
        if true_label in pred_labels[:3]:
            top3 += 1
        if true_label in pred_labels[0]:
            top1 += 1

    print()
    print("ZERO SHOT LEARNING SCORE")
    print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x_zsl))))
    print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x_zsl))))
    print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x_zsl))))


