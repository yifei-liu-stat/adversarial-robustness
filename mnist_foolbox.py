# modified from: https://foolbox.jonasrauber.de/guide/examples.html

from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import MNISTDataset

import foolbox as fb
from foolbox import PyTorchModel, accuracy
from foolbox.attacks import LinfPGD

import matplotlib.pyplot as plt


class PyNet(nn.Module):
    """CNN architecture. This is the same MNIST model from pytorch/examples/mnist repository"""
    
    def __init__(self, in_channels=1):
        super(PyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def ld_mnist():
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    
    # Load MNIST dataset
    train_dataset = MNISTDataset(root="/tmp/data", transform=train_transforms)
    test_dataset = MNISTDataset(
        root="/tmp/data", train=False, transform=test_transforms
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10000, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)





# Load training and test data
data = ld_mnist()

# Instantiate model, loss, and optimizer for training
net = PyNet(in_channels=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    net = net.cuda()

loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


# Train vanilla model
nb_epochs = 8
net.train()
for epoch in range(1, nb_epochs + 1):
    train_loss = 0.0
    for x, y in data.train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(
        "epoch: {}/{}, train loss: {:.3f}".format(
            epoch, nb_epochs, train_loss
        )
    )




# instantiate a model
net.eval()
fmodel = PyTorchModel(net, bounds=(0, 1))

# get one batch of test images from test dataloader
images, labels = next(iter(data.test))
images, labels = images.to(device), labels.to(device)

# apply the PGD attack with multiple magnitudes of variation
attack = LinfPGD()
epsilons = [.0, .0002, .0005, .0008, .001, .0015, .002, .003, .01, .1, .3, .5, 1.0]
raw_advs, adv_images, success = attack(fmodel, images, labels, epsilons=epsilons)

# test accuracy
clean_acc = accuracy(fmodel, images, labels) # test acc on clean examples
robust_acc = 1 - success.float32().mean(axis=-1) # test acc's on PGD adv examples

# visualization
clean = images[:10].to("cpu")
adv = adv_images[-5].raw[:10].to("cpu")
noise = adv - clean
tmp = torch.cat((clean, adv, noise), dim = 0)
fb.plot.images(tmp, nrows = 3)
plt.savefig("visualization.png")
# top row: clean images
# middle row: PGD adversarial images
# bottom row: perturbation