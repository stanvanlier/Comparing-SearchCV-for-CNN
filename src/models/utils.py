import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import data


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
    model.eval()
    with torch.no_grad():
        for (in_data,) in dataloader:
            output = model(in_data)
            outputs.append(output.cpu())
    return torch.cat(outputs)


def evaluate(model, batch_size, device, X, y, criterion):
    output = predict(model, batch_size, device, X)
    y = data.utils.ensure_tensors(y)[0]
    loss = criterion(output, y).item()
    accuracy = (output.argmax(1) == y).float().mean().item()
    return loss, accuracy
