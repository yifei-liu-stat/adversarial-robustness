# modified from: https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/mnist_tutorial.py

from easydict import EasyDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import MNISTDataset

from scipy.io import savemat
import copy as cp
import subprocess
import pickle
import os

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from convtoWb import convtoWb
from scipy.sparse.linalg import svds


def ld_mnist(train_batch_size = 128, test_batch_size = 1000):
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
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


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
    
    def lipschitz(self):
        """layer-wise Lipschitz constants for a giving Pynet class"""
        net = cp.deepcopy(self)
        net.to("cpu")
        net.eval()
        img_shape = (1, 1, 28, 28)
        test_img = torch.zeros(img_shape)
        conv1, _ = convtoWb(net.conv1, img_shape[1:])
        conv2, _ = convtoWb(net.conv2, net.conv1(test_img).shape[1:])
        fc1 = net.fc1.weight
        fc2 = net.fc2.weight
        Lip = [svds(s.detach().numpy(), k = 1, return_singular_vectors = False).item() for s in [conv1, conv2, fc1, fc2]]
        Lip = np.array(Lip)
        Shape = [conv1.shape, conv2.shape, fc1.shape, fc2.shape]
        return Lip, Shape
       
    def test_accuracy(self, test_loader, eps):
        net = cp.deepcopy(self)
        device = net.conv1.weight.device
        net.eval()
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)      
            x_fgm = fast_gradient_method(net, x, eps, np.inf, clip_min = 0, clip_max = 1)
            x_pgd = projected_gradient_descent(net, x, eps, 0.01, 40, np.inf, clip_min = 0, clip_max = 1)
            _, y_pred = net(x).max(1)  # model prediction on clean examples
            _, y_pred_fgm = net(x_fgm).max(1)  # model prediction on FGM adversarial examples
            _, y_pred_pgd = net(x_pgd).max(1)  # model prediction on PGD adversarial examples
            report.nb_test += y.size(0)
            report.correct += y_pred.eq(y).sum().item()
            report.correct_fgm += y_pred_fgm.eq(y).sum().item()
            report.correct_pgd += y_pred_pgd.eq(y).sum().item()
        return report.correct / report.nb_test, report.correct_fgm / report.nb_test, report.correct_pgd / report.nb_test

class FCNet(nn.Module):
    """Fully connected neural networks for training MNIST dataset"""
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)
        self.nlayers = 5
        self.fclist = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def lipschitz(self):
        """layer-wise Lipschitz constants for a given FCNet class"""
        net = cp.deepcopy(self)
        net.to("cpu")
        net.eval()
        fc1 = net.fc1.weight
        fc2 = net.fc2.weight
        fc3 = net.fc3.weight
        fc4 = net.fc4.weight
        fc5 = net.fc5.weight
        Lip = [svds(s.detach().numpy(), k = 1, return_singular_vectors = False).item() for s in [fc1, fc2, fc3, fc4, fc5]]
        Lip = np.array(Lip)
        Shape = [fc1.shape, fc2.shape, fc3.shape, fc4.shape, fc5.shape]
        return Lip, Shape
    
    def test_accuracy(self, test_loader, eps):
        net = cp.deepcopy(self)
        device = net.fc1.weight.device
        net.eval()
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_fgm = fast_gradient_method(net, x, eps, np.inf, clip_min = 0, clip_max = 1)
            x_pgd = projected_gradient_descent(net, x, eps, 0.01, 40, np.inf, clip_min = 0, clip_max = 1)
            _, y_pred = net(x).max(1)  # model prediction on clean examples
            _, y_pred_fgm = net(x_fgm).max(1)  # model prediction on FGM adversarial examples
            _, y_pred_pgd = net(x_pgd).max(1)  # model prediction on PGD adversarial examples
            report.nb_test += y.size(0)
            report.correct += y_pred.eq(y).sum().item()
            report.correct_fgm += y_pred_fgm.eq(y).sum().item()
            report.correct_pgd += y_pred_pgd.eq(y).sum().item()
        return report.correct / report.nb_test, report.correct_fgm / report.nb_test, report.correct_pgd / report.nb_test
    
    def local_lipschitz(self, x, eps):
        """Returns local Lipschitz constant for a batch of input imamges x"""
        net = cp.deepcopy(self)
        x.to("cpu")
        net.to("cpu")
        net.eval()
        x = x.flatten(1) # x.shape: (batch_size, 1, 28, 28) 
        L = net.nlayers # number of linear layers
        layer_list = net.fclist
        with torch.no_grad():
            lower, upper = x - eps, x + eps
            # lower, upper = lower.clamp(0, 1), upper.clamp(0, 1) # initialization, only clamp at input layer
            I_prev_list = [torch.eye(784) for _ in range(len(x))]
            local_lip = torch.ones(len(x))
            for l in range(L):
                # current linear layer
                fc = layer_list[l]
                # previous center and radius
                center, radius = (lower + upper) / 2, (upper - lower) / 2
                # update to current center and radius
                center = fc(center)
                radius = radius @ fc.weight.T.abs()
                # intervals after current linear layer fc
                lower, upper = center - radius, center + radius
                
                # indicator of active ReLU activation
                op_norm = torch.ones(len(x))
                I_current_list = cp.deepcopy(I_prev_list)
                for i in range(len(x)):
                    I_prev = I_prev_list[i]
                    I_current = (upper[i] > 0).type(torch.float32).diag() if l < (L - 1) else torch.eye(10)
                    W = I_current @ fc.weight @ I_prev
                    
                    op_norm[i] = svds(W.detach().numpy(), k = 1, return_singular_vectors = False).item()
                    I_current_list[i] = I_current
                I_prev_list = cp.deepcopy(I_current_list)
                
                # update local Lipschitz constant
                local_lip = local_lip * op_norm
        return local_lip

def LipSDP(mynet):
    """Returns global Lipschitz constant of FCN class based on LipSDP method"""
    # from a FCN class
    net = cp.deepcopy(mynet)
    net.to("cpu")
    net.eval()
    weights = []
    for i in range(5):
        keyword = "fc" + np.array2string(np.array(i + 1)) + ".weight"
        weights.append(np.float64(net.state_dict()[keyword].numpy()))
    
    fname = os.path.join(os.getcwd(), 'data/weights/random_weights.mat')
    data = {'weights': np.array(weights, dtype=np.object)}
    savemat(fname, data)
    
    script_path = os.path.join(os.getcwd(), 'test_LipSDP.sh')
    with open(script_path, 'rb') as file:
        script = file.read()
    
    subprocess.call(script, shell = True)
    L = pickle.load(open("data/LipSDP.pkl", "rb"))
    return L


def samples_partition(net, whole_test_loader, attack = "none", eps = 0.05):
    """Returns correctly & incorrectly classified samples (either standard or adversarial) by net"""
    net = cp.deepcopy(net)
    net.to("cpu")
    net.eval()
    x_ori, y_ori = next(iter(whole_test_loader))
    x_ori, y_ori = x_ori.to("cpu"), y_ori.to("cpu")
    
    if attack == "none":
        x_new = cp.deepcopy(x_ori)
    elif attack == "fgsm":
        x_new = fast_gradient_method(net, x_ori, eps, np.inf, clip_min = 0, clip_max = 1)
    elif attack == "pgd":
        x_new = projected_gradient_descent(net, x_ori, eps, 0.01, 40, np.inf, clip_min = 0, clip_max = 1)
    else:
        raise ValueError('attack is expected to be \"none\", \"fgsm\" or \"pgd\"')
    
    _, y_pred = net(x_new).max(1)
    correct = x_ori[y_pred.eq(y_ori)]
    incorrect = x_ori[~y_pred.eq(y_ori)]
    return correct, incorrect




# Load training and test data
data = ld_mnist()

# net = PyNet(in_channels=1) 
net = FCNet()

# torch.cuda.set_device(7) # 0 for default cuda device cuda:0
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    net = net.cuda()

loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


# Train vanilla model
nb_epochs = 5
eps = 0.2 # total epsilon for FGSM and PGD attacks
adv_train = True # whether to use PGD adversarial training
adv_method = "PGD" # "PGD", method of adversarial training when adv_train = True; alternative: "FGSM"
detail = False # whether to save detailed information at each epoch
save_model = False # whether to save the model at the end of training
global_lip = True

if global_lip:
    LipSDP_list = np.zeros(nb_epochs)

if detail:
    Lip_list = np.zeros((nb_epochs, 4 if isinstance(net, PyNet) else 5))
    clean_acc_list = np.zeros(nb_epochs)
    fgsm_acc_list = np.zeros(nb_epochs)
    pgd_acc_list = np.zeros(nb_epochs)

net.train()
for epoch in range(1, nb_epochs + 1):
    train_loss = 0.0
    for x, y in data.train:
        x, y = x.to(device), y.to(device)
        if adv_train:
            # Replace clean example with adversarial example for adversarial training
            if adv_method == "PGD":
                x = projected_gradient_descent(net, x, eps, 0.01, 40, np.inf, clip_min = 0, clip_max = 1)
            elif adv_method == "FGSM":
                x = fast_gradient_method(net, x, eps, np.inf, clip_min = 0, clip_max = 1)
        optimizer.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if global_lip:
        LipSDP_list[epoch - 1] = LipSDP(net)    
    if detail:
        # Calculate layer-wise Lipschitz constants
        Lip_list[epoch-1, :], _ = net.lipschitz()
        clean_acc_list[epoch-1], fgsm_acc_list[epoch-1], pgd_acc_list[epoch-1] = net.test_accuracy(data.test, eps)
    print(
        "epoch: {}/{}, train loss: {:.3f}".format(
            epoch, nb_epochs, train_loss
        )
    )



# string information
if adv_train:
    if adv_method == "PGD":
        adv_string = "pgd"
    elif adv_method == "FGSM":
        adv_string = "fgsm"
else:
    adv_string = "standard"

net_string = "cnn" if isinstance(net, PyNet) else "fnn"

tmp = np.array(eps)
tmp = np.array2string(tmp)
eps_string = tmp.replace(".", "_")

epoch_string = "epoch" + np.array2string(np.array(nb_epochs))

# save detailed training information at each epoch when detail = True
if detail:    
    save_path = "_".join(["data/mnist", adv_string, net_string, eps_string]) + ".pkl"
    result = [Lip_list, clean_acc_list, fgsm_acc_list, pgd_acc_list]
    pickle.dump(result, open(save_path, "wb"))

# save model
if save_model:
    save_path = "_".join(["model/mnist", adv_string, net_string, eps_string]) + ".pt"
    torch.save(net, save_path)





# local Lipschitz constant with ReLU patterns and interval bound propagation

## distributions of loc-Lip's for different kinds of FNN (trained 20 epochs)
testloader = ld_mnist(test_batch_size = 10000).test
x, _ = next(iter(testloader))

eps = 0.01
tmp = np.array(eps)
tmp = np.array2string(tmp)
eps_string = tmp.replace(".", "_")


model_standard = torch.load("model/mnist_standard_fnn_" + eps_string + ".pt")
model_standard.eval()
local_lip_standard = model_standard.local_lipschitz(x, eps = eps)
naive_lip_standard = model_standard.lipschitz()[0].prod()
global_lip_standard = LipSDP(model_standard)

model_fgsm = torch.load("model/mnist_fgsm_fnn_" + eps_string + ".pt")
model_fgsm.eval()
local_lip_fgsm = model_fgsm.local_lipschitz(x, eps = eps)
naive_lip_fgsm = model_fgsm.lipschitz()[0].prod()
global_lip_fgsm = LipSDP(model_fgsm)

model_pgd = torch.load("model/mnist_pgd_fnn_" + eps_string + ".pt")
model_pgd.eval()
local_lip_pgd = model_pgd.local_lipschitz(x, eps = eps)
naive_lip_pgd = model_pgd.lipschitz()[0].prod()
global_lip_pgd = LipSDP(model_pgd)


result = {
    "local_lip_standard": local_lip_standard,
    "naive_lip_standard": naive_lip_standard,
    "global_lip_standard": global_lip_standard,
    "local_lip_fgsm": local_lip_fgsm,
    "naive_lip_fgsm": naive_lip_fgsm,
    "global_lip_fgsm": global_lip_fgsm,
    "local_lip_pgd": local_lip_pgd,
    "naive_lip_pgd": naive_lip_pgd,
    "global_lip_pgd": global_lip_pgd
}

save_path = "data/three_lip_for_three_models_" + eps_string + ".pkl"
# pickle.dump(result, open(save_path, "wb"))
result = pickle.load(open(save_path, "rb"))

kwargs = {"bins": 20, "density": True, "alpha": 0.4, "edgecolor": "black"}
plt.hist(result["local_lip_standard"].numpy(), label = "adam", color = "blue", **kwargs)
plt.hist(result["local_lip_fgsm"].numpy(), label = "fgsm", color = "orange", **kwargs)
plt.hist(result["local_lip_pgd"].numpy(), label = "pgd", color = "green", **kwargs)
plt.axvline(result["naive_lip_standard"], color = "blue", linestyle = "--")
plt.axvline(result["naive_lip_fgsm"], color = "orange", linestyle = "--")
plt.axvline(result["naive_lip_pgd"], color = "green", linestyle = "--")
plt.axvline(result["global_lip_standard"], color = "blue", linestyle = ":")
plt.axvline(result["global_lip_fgsm"], color = "orange", linestyle = ":")
plt.axvline(result["global_lip_pgd"], color = "green", linestyle = ":")
plt.legend(loc = "best")
plt.xlabel("Local Lipschitz constant (upper bound)")
plt.ylabel("density")
plt.title("Distributions of local lipschitz constant (eps = " + tmp + ")")
plt.savefig("figure/local_lip_dist_3nn_" + eps_string + ".png", bbox_inches = "tight")
plt.close()





## distributions of loc-Lip's for robust examples and non-robust examples
data = ld_mnist(test_batch_size = 10000)
testloader = data.test

model_standard = torch.load("model/mnist_standard_fnn_0_05.pt")
model_standard.eval()

### FGSM attacked images
x_c, x_ic = samples_partition(model_standard, testloader, "fgsm", 0.05)
local_lip_standard_correct_fgsm = model_standard.local_lipschitz(x_c, eps = 0.05)
local_lip_standard_incorrect_fgsm = model_standard.local_lipschitz(x_ic, eps = 0.05)

kwargs = {"bins": 20, "density": True, "alpha": 0.5, "edgecolor": "black"}
plt.hist(local_lip_standard_correct_fgsm.numpy(), label = "robust", color = "blue", **kwargs)
plt.hist(local_lip_standard_incorrect_fgsm.numpy(), label = "susceptible", color = "orange", **kwargs)
plt.legend(loc = "best")
plt.xlabel("Local Lipschitz constant (upper bound)")
plt.ylabel("density")
plt.title("Distributions of local lipschitz constant (FGSM attack)")
plt.savefig("figure/local_lip_dist_c_ic_fgsm.png")
plt.close()


### PGD attacked images
x_c, x_ic = samples_partition(model_standard, testloader, "pgd", 0.05)
local_lip_standard_correct_pgd = model_standard.local_lipschitz(x_c, eps = 0.05)
local_lip_standard_incorrect_pgd = model_standard.local_lipschitz(x_ic, eps = 0.05)

kwargs = {"bins": 20, "density": True, "alpha": 0.5, "edgecolor": "black"}
plt.hist(local_lip_standard_correct_pgd.numpy(), label = "robust", color = "blue", **kwargs)
plt.hist(local_lip_standard_incorrect_pgd.numpy(), label = "susceptible", color = "orange", **kwargs)
plt.legend(loc = "best")
plt.xlabel("Local Lipschitz constant (upper bound)")
plt.ylabel("density")
plt.title("Distributions of local lipschitz constant (PGD attack)")
plt.savefig("figure/local_lip_dist_c_ic_pgd.png")
plt.close()


result = {
    "local_lip_standard_correct_fgsm": local_lip_standard_correct_fgsm,
    "local_lip_standard_incorrect_fgsm": local_lip_standard_incorrect_fgsm,
    "local_lip_standard_correct_pgd": local_lip_standard_correct_pgd,
    "local_lip_standard_incorrect_pgd": local_lip_standard_incorrect_pgd
}

save_path = "data/loc_lip_for_correct_incorrect_0_05.pkl"
# pickle.dump(result, open(save_path, "wb"))
result = pickle.load(open(save_path, "rb"))




# Global-Lip as eps grows, and compared to naive bound






# visualization of adv trained CNN (20 epochs) on different lev
eps_list = np.arange(0, 0.4001, 0.025)

model_standard = torch.load("model/mnist_standard_cnn_0_25.pt")
model_standard.eval()
result_standard = np.zeros((3, len(eps_list)))
for i, test_eps in enumerate(eps_list, 0):
    result_standard[:, i] = model_standard.test_accuracy(data.test, test_eps)
    print("round:", i)

model_fgsm = torch.load("model/mnist_fgsm_cnn_0_25.pt")
model_fgsm.eval()
result_fgsm = np.zeros((3, len(eps_list)))
for i, test_eps in enumerate(eps_list, 0):
    result_fgsm[:, i] = model_fgsm.test_accuracy(data.test, test_eps)
    print("round:", i)

model_pgd = torch.load("model/mnist_pgd_cnn_0_25.pt")
model_pgd.eval()
result_pgd = np.zeros((3, len(eps_list)))
for i, test_eps in enumerate(eps_list, 0):
    result_pgd[:, i] = model_pgd.test_accuracy(data.test, test_eps)
    print("round:", i)

## comparision on fgsm trained cnn
plt.plot(eps_list, result_standard[1, :], 'b.--', label = "fgsm_adam")
plt.plot(eps_list, result_standard[2, :], 'r.--', label = "pgd_adam")
plt.plot(eps_list, result_fgsm[1, :], 'b.-', label = "fgsm_fgsm")
plt.plot(eps_list, result_fgsm[2, :], 'r.-', label = "pgd_fgsm")
plt.axvline(x = 0.25, color = "k", linestyle = "--")
plt.legend()
plt.xlabel("eps")
plt.ylabel("test accuracy")
plt.title("Accuracy of FGSM trained CNN on adversarial attacks")
plt.savefig("figure/mnist_fgsm_cnn_epss.png")
plt.close()

## comparision on pgd trained cnn
plt.plot(eps_list, result_standard[1, :], 'b.--', label = "fgsm_adam")
plt.plot(eps_list, result_standard[2, :], 'r.--', label = "pgd_adam")
plt.plot(eps_list, result_pgd[1, :], 'b.-', label = "fgsm_pgd")
plt.plot(eps_list, result_pgd[2, :], 'r.-', label = "pgd_pgd")
plt.axvline(x = 0.25, color = "k", linestyle = "--")
plt.legend()
plt.xlabel("eps")
plt.ylabel("test accuracy")
plt.title("Accuracy of PGD trained CNN on adversarial attacks")
plt.savefig("figure/mnist_pgd_cnn_epss.png")
plt.close()

    



# visualization of clean and adversarial image
# 0.3 as radius of L_infty ball seems to large, 0.02 seems a reasonable one

net.eval()
x, _ = next(iter(data.train))
clean_image = x[1, None]
pgd_image = projected_gradient_descent(net.to("cpu"), clean_image, 0.02, 0.01, 40, np.inf, clip_min = 0, clip_max = 1).detach()


fig, axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].imshow(clean_image[0][0])
axes[1].imshow(pgd_image[0][0])
plt.savefig("figure/temp.png")
plt.close()




# visualization --- CNN
training_standard = pickle.load(open("data/mnist_standard_cnn.pkl", "rb"))
training_pgd = pickle.load(open("data/mnist_pgd_cnn.pkl", "rb"))
epoch = np.arange(1, 101)

## layer-wise Lip
plt.plot(epoch, training_standard[0][:, 0], 'y-', linewidth = 1, label = "conv1_adam")
plt.plot(epoch, training_standard[0][:, 1], 'g-', linewidth = 1, label = "conv2_adam")
plt.plot(epoch, training_standard[0][:, 2], 'b-', linewidth = 1, label = "fc1_adam")
plt.plot(epoch, training_standard[0][:, 3], 'r-', linewidth = 1, label = "fc2_adam")
plt.plot(epoch, training_pgd[0][:, 0], 'y--', linewidth = 1, label = "conv1_pgd")
plt.plot(epoch, training_pgd[0][:, 1], 'g--', linewidth = 1, label = "conv2_pgd")
plt.plot(epoch, training_pgd[0][:, 2], 'b--', linewidth = 1, label = "fc1_pgd")
plt.plot(epoch, training_pgd[0][:, 3], 'r--', linewidth = 1, label = "fc2_pgd")
plt.legend(loc = "best", ncol = 2)
plt.xlabel("epoch")
plt.ylabel("Lipschitz constant")
plt.title("Comparision of layer-wise Lipschitz constant")
plt.savefig("figure/layerwise_cnn.png")
plt.close()

## accuracy
plt.plot(epoch, training_standard[1], 'g-', linewidth = 1, label = "clean_adam")
plt.plot(epoch, training_standard[2], 'b-', linewidth = 1, label = "fgsm_adam")
plt.plot(epoch, training_standard[3], 'r-', linewidth = 1, label = "pgd_adam")
plt.plot(epoch, training_pgd[1], 'g--', linewidth = 1, label = "clean_pgd")
plt.plot(epoch, training_pgd[2], 'b--', linewidth = 1, label = "fgsm_pgd")
plt.plot(epoch, training_pgd[3], 'r--', linewidth = 1, label = "pgd_pgd")
plt.legend(loc = "best", ncol = 2)
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.title("Test accuracy of standard/robust trained CNN")
plt.savefig("figure/accuracy_cnn.png")
plt.close()

## standard training on adv accuracy
training_standard = pickle.load(open("data/mnist_standard_cnn_0_1.pkl", "rb"))
epoch = np.arange(1, 101)

plt.plot(epoch, training_standard[1], 'g-', linewidth = 1, label = "clean")
plt.plot(epoch, training_standard[2], 'b-', linewidth = 1, label = "fgsm")
plt.plot(epoch, training_standard[3], 'r-', linewidth = 1, label = "pgd")
plt.legend(loc = "best")
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.title("Test accuracy of standard trained CNN")
plt.savefig("figure/accuracy_cnn_0_1.png")
plt.close()





# visualization --- FNN

training_standard = pickle.load(open("data/mnist_standard_fnn.pkl", "rb"))
training_pgd = pickle.load(open("data/mnist_pgd_fnn.pkl", "rb"))
epoch = np.arange(1, 101)

## layer-wise Lip
plt.plot(epoch, training_standard[0][:, 0], 'y-', linewidth = 1, label = "fc1_adam")
plt.plot(epoch, training_standard[0][:, 1], 'g-', linewidth = 1, label = "fc2_adam")
plt.plot(epoch, training_standard[0][:, 2], 'b-', linewidth = 1, label = "fc3_adam")
plt.plot(epoch, training_standard[0][:, 3], 'r-', linewidth = 1, label = "fc4_adam")
plt.plot(epoch, training_standard[0][:, 4], 'k-', linewidth = 1, label = "fc5_adam")

plt.plot(epoch, training_pgd[0][:, 0], 'y--', linewidth = 1, label = "fc1_pgd")
plt.plot(epoch, training_pgd[0][:, 1], 'g--', linewidth = 1, label = "fc2_pgd")
plt.plot(epoch, training_pgd[0][:, 2], 'b--', linewidth = 1, label = "fc3_pgd")
plt.plot(epoch, training_pgd[0][:, 3], 'r--', linewidth = 1, label = "fc4_pgd")
plt.plot(epoch, training_pgd[0][:, 4], 'k--', linewidth = 1, label = "fc5_pgd")

plt.legend(loc = "best", ncol = 2)
plt.xlabel("epoch")
plt.ylabel("Lipschitz constant")
plt.title("Comparision of layer-wise Lipschitz constant")
plt.savefig("figure/layerwise_fnn.png")
plt.close()

## accuracy
plt.plot(epoch, training_standard[1], 'g-', linewidth = 1, label = "clean_adam")
plt.plot(epoch, training_standard[2], 'b-', linewidth = 1, label = "fgsm_adam")
plt.plot(epoch, training_standard[3], 'r-', linewidth = 1, label = "pgd_adam")
plt.plot(epoch, training_pgd[1], 'g--', linewidth = 1, label = "clean_pgd")
plt.plot(epoch, training_pgd[2], 'b--', linewidth = 1, label = "fgsm_pgd")
plt.plot(epoch, training_pgd[3], 'r--', linewidth = 1, label = "pgd_pgd")
plt.legend(loc = "best", ncol = 2)
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.title("Test accuracy of standard/robust trained FNN")
plt.savefig("figure/accuracy_fnn.png")
plt.close()

## standard training on adv accuracy
training_standard = pickle.load(open("data/mnist_standard_fnn_0_1.pkl", "rb"))
epoch = np.arange(1, 101)

plt.plot(epoch, training_standard[1], 'g-', linewidth = 1, label = "clean")
plt.plot(epoch, training_standard[2], 'b-', linewidth = 1, label = "fgsm")
plt.plot(epoch, training_standard[3], 'r-', linewidth = 1, label = "pgd")
plt.legend(loc = "best")
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.title("Test accuracy of standard trained FNN")
plt.savefig("figure/accuracy_fnn_0_1.png")
plt.close()




