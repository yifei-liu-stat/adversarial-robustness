import sys
sys.path.insert(0, '/home/liu00980/Documents/8980project/local-lipschitz')

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle
from datetime import datetime

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F

import data_load
import utils
import Local_bound as Local


args = utils.argparser(data='mnist',epochs=300,warmup=0,rampup=150,batch_size=256,epsilon=1.58,epsilon_train=1.58)    
train_loader, test_loader = data_load.data_loaders(args.data, args.batch_size, args.test_batch_size, augmentation=args.augmentation, normalization=args.normalization, drop_last=args.drop_last, shuffle=args.shuffle)
model = utils.select_model(args.data, "relux", args.init)
net_local = Local.net_Local_Lip(model, args)


epsilon1 = 1.0
X, y, idx = next(iter(train_loader))
X, y = X.cuda(), y.cuda()

# compute the feature size at each layer
input_size = []
depth = len(model)
x = torch.randn(1,1,28,28).cuda()
for i, layer in enumerate(model.children()):
    if i < depth-1:
        input_size.append(x.size()[1:])
        x = layer(x)

# create u on cpu to store singular vector for every input at every layer
u_train = []
u_test = []

# only linear layer will be inspected (conv layer, fc layer)
for i in range(len(input_size)):
    print(i)
    if not model[i].__class__.__name__=='ReLU_x' and not model[i].__class__.__name__=='Flatten':
        u_train.append(torch.randn((len(train_loader.dataset), *(input_size[i])), pin_memory=True))
        u_test.append(torch.randn((len(test_loader.dataset), *(input_size[i])), pin_memory=True))
    else:
        u_train.append(None)
        u_test.append(None)


# extract singular vector for each data point ON EACH BATCH indexed by idx
u_train_data = []
for ll in range(len(u_train)):
    if u_train[ll] is not None:
        u_train_data.append(u_train[ll][idx,:].cuda())
    else:
        u_train_data.append(None)


u_list = None
_, _, r_prev, _, _, _, _, _, _ = net_local(X, 0.3, u_list, u_train_data, 10)










