import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import data

import os
from glob import glob

from collections import OrderedDict

# torch does only have mappings torch.<Type>Storage => torch.dtype, not
# the other way around, so we create a torch.dtype => torch.<Type>Storage.
dtype2TypedStorage = {ts.dtype: ts
                      for ts in torch.storage._TypedStorage.__subclasses__()
                      # filter only cpu storage
                      if ts.__module__ == 'torch'}

def init_memory_mapped_tensor(path, shape, dtype):
    """Iinitialize a new empty memory mapped tensor"""
    shape_str = "x".join(str(a) for a in shape)
    TS = dtype2TypedStorage[dtype]
    dtype_str = str(dtype).split('.')[-1]
    fname = f"{path}.{shape_str}.{dtype_str}"
    shape = torch.Size(shape)
    disk_storage = TS.from_file(fname, True, shape.numel())
    return torch.tensor(disk_storage, dtype=dtype).reshape(shape)

def to_memory_mapped_tensor(path, tensor):
    """Copies tensor to disk, and returns the new memory mapped version"""
    mmaptensor = init_memory_mapped_tensor(path, tensor.shape, tensor.dtype)
    return mmaptensor.copy_(tensor)

def load_memory_mapped_tensor(fname):
    assert os.path.isfile(fname)
    *path, shape_str, dtype_str = fname.split(".")
    dtype = getattr(torch, dtype_str.lower())  # get dtype from torch module (only cpu variants are directly in torch)
    assert dtype in dtype2TypedStorage
    if shape_str == '':
        shape = torch.Size()
    else:
        shape = torch.Size(int(a) for a in shape_str.split("x"))
    path = ".".join(path)
    return init_memory_mapped_tensor(path, shape, dtype)

def load_memory_mapped_tensor_by_pattern(pattern):
    match = glob(pattern)
    if not match:
        raise FileNotFoundError(pattern)
    assert len(match) == 1 # check if pattern not ambiguous
    fname = match[0]
    return load_memory_mapped_tensor(fname)

# TODO make function nested_tensors2flat_tensor_with_index
def flat_tensor2nested_tensors(flat_tensor, index):
    """handy to make a list of memory mapped tensors (from the same file) having varying lenghts"""
    return [flat_tensor[index[i]:index[i+1]] for i in range(len(index)-1)]


class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(tensor.to(self.device) for tensor in batch)


def wrapped_device_data_loader(dataloader, device):
    if 'xla' in str(device):
        import torch_xla.distributed.parallel_loader as pl
        dataloader = pl.MpDeviceLoader(dataloader, device)
    elif 'cuda' in str(device):
        dataloader = DeviceDataLoader(dataloader, device)
    return dataloader


def train(model, optimizer, criterion, epochs, batch_size, device, X, y):
    X, y = data.utils.ensure_tensors(X, y)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = wrapped_device_data_loader(
        torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, drop_last=False),
        device)

    model.to(device)
    model.train()
    losses = []
    accuracies = []
    total_samples = 0
    for e in range(epochs):
        for (in_data, target) in dataloader:
            optimizer.zero_grad()
            output = model(in_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_len = len(target)
            total_samples += batch_len
            losses.append(loss.item()*batch_len)
            accuracies.append(
                (output.detach().argmax(1) == target).float().mean().item()*batch_len)
    optimizer.zero_grad()
    #model.cpu()
    model.mmap()
    loss = sum(losses)/total_samples
    accuracy = sum(accuracies)/total_samples
    return loss, accuracy


def predict(model, batch_size, device, X):
    X = data.utils.ensure_tensors(X)[0]
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = wrapped_device_data_loader(
        torch.utils.data.DataLoader(dataset, batch_size=batch_size),
        device)
    outputs = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (in_data,) in dataloader:
            output = model(in_data)
            outputs.append(output.cpu())
    #model.cpu()
    model.mmap()
    return torch.cat(outputs)


def evaluate(model, batch_size, device, X, y, criterion):
    output = predict(model, batch_size, device, X)
    y = data.utils.ensure_tensors(y)[0]
    with torch.no_grad():
        loss = criterion(output, y).item()
        accuracy = (output.argmax(1) == y).float().mean().item()
    return loss, accuracy
