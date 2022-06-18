import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

class CNN(nn.Module):
    def __init__(self, mmap_path=None, n_conv_layers=2, first_channels=1, last_channels=20,
                 first_kernel_size=5, last_kernel_size=3, n_linear_layers=2, n_classes=10, 
                 pooling='MaxPool2d', pool_size=2, activation='ReLU', conv_order="pan"):
        super(CNN, self).__init__()

        convs_channels = torch.linspace(first_channels, last_channels, n_conv_layers+1, 
                                        dtype=torch.int).tolist()
        convs_kernel_size = torch.linspace(first_kernel_size, last_kernel_size, n_conv_layers, 
                                           dtype=torch.int).tolist()

        Pooling = getattr(torch.nn, pooling)
        Activation = getattr(torch.nn, activation)
        
        after_conv_constructor = {
            "p": lambda _: Pooling(pool_size),
            "a": lambda _: Activation(),
            "n": lambda out_channels: nn.BatchNorm2d(out_channels),
        }

        self.convs = nn.Sequential()
        for li in range(n_conv_layers):
            in_channels = convs_channels[li]
            out_channels = convs_channels[li+1]
            kernel_size = convs_kernel_size[li] 

            self.convs.append( nn.Conv2d(in_channels, out_channels, kernel_size) )
            for c in conv_order:
                self.convs.append( after_conv_constructor[c](out_channels) )

        linears_features = torch.linspace(last_channels, n_classes, n_linear_layers+1, 
                                          dtype=torch.int).tolist()
        self.linears = nn.Sequential()
        for li in range(n_linear_layers):
            in_features = linears_features[li]
            out_features = linears_features[li+1]

            self.linears.append( nn.Linear(in_features, out_features) )
            if li != n_linear_layers-1:
                self.linears.append( Activation() )

        if mmap_path is None:
            mmap_path = 'mmap/' + datetime.now().strftime("%Y%m%dT%H%M%S.%f")
        self.mmap_path = mmap_path

    def mmap(self, mmap_path=None):
        if mmap_path is None:
            mmap_path  = self.mmap_path
        os.makedirs(mmap_path, exist_ok=True)
        self.cpu()
        for n, p in self.named_parameters():
            p.data = utils.to_memory_mapped_tensor(f'{mmap_path}/{n}', p.data)

    def forward(self, x):
        x = self.convs(x)
        # take mean over width and heigth (instead of flatten to not depend on input size)
        x = x.mean(dim=[2,3])
        x = self.linears(x)
        return x

# Only instantiate model weights once in memory.
# Not needed since we want to train multiple models
# WRAPPED_MODEL = xmp.MpModelWrapper(MNIST())

