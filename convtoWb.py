# Convert a convolutional layer to a Toeplitz matrix with two approaches

import numpy as np
from scipy import linalg
import time

import torch
import torch.nn.functional as F
import torchvision.models as models

from scipy.sparse.linalg import svds

# A nice and elegant way, but targeted for padding = 1, stride = 1 (can't customized)
# credit: https://stackoverflow.com/questions/60643786/2d-convolution-with-padding-same-via-toeplitz-matrix-multiplication?noredirect=1&lq=1

def toeplitz_1_ch(kernel, input_size):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    #o_h, o_w = i_h-k_h+1, i_w-k_w+1
    o_h, o_w = i_h, i_w
    # construct 1d conv toeplitz matrices for the kernel, with "same" padding
    n = i_h
    
    K1 = np.zeros((n,))
    K1[:2] = (kernel[1,1], kernel[1,2] )
    K2 = np.zeros((n,))
    K2[:2] = (kernel[1,1], kernel[1,0])
    
    K = linalg.toeplitz(c=K2, r = K1)
    KK = np.identity(n)
    
    L1 = np.zeros((n,))
    L1[:2] = (kernel[2,1], kernel[2,2])
    L2 = np.zeros((n,))
    L2[:2] = (kernel[2,1], kernel[2,0])
    
    t = np.zeros(n)
    s = np.zeros(n)
    s[1] = 1
    L = linalg.toeplitz(c = L2, r = L1)
    LL = linalg.toeplitz(r = s, c = t)
    
    A = np.kron(LL, L) + np.kron(KK, K)
    
    L1 = np.zeros((n,))
    L1[:2] = (kernel[0,1], kernel[0,2])
    L2 = np.zeros((n,))
    L2[:2] = (kernel[0,1], kernel[0,0])
    
    L = linalg.toeplitz(c = L2, r = L1)
    LL = linalg.toeplitz(c = s, r = t)
    A = A + np.kron(LL, L)
    return A


def toeplitz_mult_ch(kernel, input_size):
    """Compute toeplitz matrix for 2d conv with multiple in and out channels.
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input_size: (n_in, H_i, W_i)
    
    Returns:
        T: Toeplitz matrix corresponding to convolutional layer with stride = 1, padding = 1, bias = 0
    
    Examples:
        >>> k = np.random.randn(4*3*3*3).reshape((4,3,3,3))
        >>> i = np.random.randn(3,9,9)
        >>> T = toeplitz_mult_ch(k, i.shape)
        >>> out = T.dot(i.flatten()).reshape((1,4,9,9))
        >>> 
        >>> # check correctness of convolution via toeplitz matrix
        >>> print(np.sum((out - F.conv2d(torch.tensor(i).view(1,3,9,9), torch.tensor(k), padding = 1).numpy())**2))
        >>> 
        >>> # work with torch.nn.module
        >>> conv = torch.nn.Conv2d(1, 2, 3, 1)
        >>> print("shape of the kernel:", conv.weight.shape) # (c_output, c_input, k_height, k_width)
        >>> img = torch.rand((1, 4, 4))
        >>> W = toeplitz_mult_ch(conv.weight, img.shape)
        >>> out_W = W.dot(img.numpy().flatten())
        >>> out_conv = conv(img.reshape(1, 1, 4, 4)).flatten()
    """
    
    
    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1], input_size[2])
    T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))
    
    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            T[i, :, j, :] = T_k
    T.shape = (np.prod(output_size), np.prod(input_size))
    
    return T









# A quick but dirty way (use the fact that CONV operation is a linear operator)

# # credit: https://github.com/pytorch/pytorch/issues/26781#issuecomment-749777861
# def to_matrix(conv, image_shape):
#     identity = torch.eye(np.prod(image_shape).item()).reshape([-1] + list(image_shape))
#     output = F.conv2d(identity, conv.weight, None, conv.stride, conv.padding)
#     output_shape = output.shape[1:]
#     W = output.reshape(-1, np.prod(output_shape).item()).T
#     b = torch.stack([torch.ones(output_shape[1:]) * bi for bi in conv.bias])
#     b = b.reshape(-1, np.prod(output_shape).item())
#     return W, b



def convtoWb(conv, img_shape):
    """Convert convolutional operation to matrix-vector multiplication with bias
    as in CONV(img) = W % vec(img) + bias
    
    Args:
        conv (torch.nn.modules.conv.Conv2d): convolutional layer defined by torch.nn.Conv2d()
        img_shape (tuple): a tuple (C, H, W) indicating the shape of an image
    
    Returns:
        W (torch.Tensor): weight matrix of shape (output_dim, input_dim)
        b (torch.Tensor): bias vector of shape (output_dim, )
        
    Examples:
        >>> conv = torch.nn.Conv2d(3, 2, 3)
        >>> img_shape = (3, 6, 6)
        >>> W, b = convtoWb(conv, img_shape)
        >>> print(W)
        >>> print(b)
        >>>
        >>> # check the correctness
        >>> img = torch.rand(img_shape)
        >>> out1 = conv(img.reshape(1, *img.shape)).flatten()
        >>> out2 = torch.matmul(W, img.flatten()) + b
        >>> print((out1 - out2).abs().max().item())    
    """
    dim_in = np.prod(img_shape)
    img_tensor = torch.eye(dim_in).reshape(dim_in, *img_shape)
    out_tensor = F.conv2d(img_tensor, conv.weight, None, conv.stride, conv.padding, conv.dilation) # disable bias
    W = out_tensor.flatten(1).T
    b = torch.cat([torch.ones(np.prod(out_tensor.shape[2:])) * bias for bias in conv.bias])
    return W, b


if __name__ == "main":
    # layer-wise Lipschitz constant of pretrained alexnet
    alexnet = models.alexnet(pretrained = True)
    alexnet.eval()
    alexnet.to("cpu")


    feature_index = [0, 3, 6, 8, 10]
    classifier_index = [1, 4, 6]
    img_shape = (1, 3, 224, 224)
    test_img = torch.zeros(img_shape)


    feature_W = []
    for i in feature_index:
        tmp_img = alexnet.features[:i](test_img)
        conv = alexnet.features[i]
        W, _ = convtoWb(conv, tmp_img.shape[1:])
        feature_W = feature_W + [W]

    classifier_W = []
    for i in classifier_index:
        W = alexnet.classifier[i].weight
        classifier_W = classifier_W + [W]


    def svds_timing(W, ndigits = 4):
        epsilon = 1 / 10**(ndigits + 1)
        begin = time.time()
        sigma1 = svds(W.detach().numpy(), k = 1, tol = epsilon, return_singular_vectors = False).item()
        seconds = time.time() - begin
        shape = W.shape
        return {"lipschitz": round(sigma1, ndigits), "shape": shape, "time": round(seconds, ndigits)}

    Lip = [svds_timing(s) for s in feature_W + classifier_W]
    for result in Lip:
        print(result)

    # import pickle
    # pickle.dump(Lip, open("data/lipschitz_alex.pkl", "wb"))
    # Lip = pickle.load(open("data/lipschitz_alex.pkl", "rb"))


