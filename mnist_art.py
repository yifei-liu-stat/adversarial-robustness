# modified from: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist


# CNN for trainig MNIST
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)
    
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

# load dataset and swap axes to PyTorch's NCHW format
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)


# train vanilla model
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# ART wrapper
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)


# prediction accuracy
pred_clean = classifier.predict(x_test)

attack_fsgm = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv_fsgm = attack_fsgm.generate(x=x_test)
pred_fsgm = classifier.predict(x_test_adv_fsgm)

attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=0.2)
x_test_adv_pgd = attack_pgd.generate(x=x_test) # take a while
pred_pgd = classifier.predict(x_test_adv_pgd)

acc_clean = np.sum(np.argmax(pred_clean, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
acc_fsgm = np.sum(np.argmax(pred_fsgm, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
acc_pgd = np.sum(np.argmax(pred_pgd, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)



x = F.max_pool2d(x, 2, 2)
x = F.relu(self.conv_2(x))
x = F.max_pool2d(x, 2, 2)




import torch
from convtoWb import convtoWb
from scipy.sparse.linalg import svds

# layer-wise Lipschitz constant and naive global Lipschitz constant
model.eval()
model.to("cpu")

img_shape = (1, 1, 28, 28)
test_img = torch.rand(img_shape)
conv1, _ = convtoWb(model.conv_1, img_shape[1:])
conv2, _ = convtoWb(model.conv_2, F.max_pool2d(model.conv_1(test_img), 2, 2).shape[1:])
fc1 = model.fc_1.weight
fc2 = model.fc_2.weight

# largest singular value for sparse matrix
Lip = [svds(s.detach().numpy(), k = 1, return_singular_vectors = False).item() for s in [conv1, conv2, fc1, fc2]]
Lip
