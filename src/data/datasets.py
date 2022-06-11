import torchvision
import torch
import numpy as np

def MNIST(asnumpy=False, PIL_friendly=False):
    train_ds = torchvision.datasets.MNIST("downloads", train=True, download=True)
    test_ds = torchvision.datasets.MNIST("downloads", train=False, download=True)
    if PIL_friendly:
        traindata = train_ds.data
        testdata = test_ds.data
    else:
        traindata = train_ds.data.unsqueeze(1)
        testdata = test_ds.data.unsqueeze(1)
    if asnumpy:
        return ( (traindata.numpy(), train_ds.targets.numpy()), 
                 (testdata.numpy(), test_ds.targets.numpy())    )
    return (traindata, train_ds.targets), (testdata, test_ds.targets)

def SVHN(asnumpy=False, PIL_friendly=False):
    train_ds = torchvision.datasets.SVHN("downloads/SVHN", split='train', download=True)
    test_ds = torchvision.datasets.SVHN("downloads/SVHN", split='test', download=True)
    if PIL_friendly:
        traindata = train_ds.data.transpose((0,2,3,1))
        testdata = test_ds.data.transpose((0,2,3,1))
    else:
        traindata = train_ds.data
        testdata = test_ds.data
    if asnumpy:
        return (traindata, train_ds.labels), (testdata, test_ds.labels)
    else:
        return ( (torch.from_numpy(traindata), torch.from_numpy(train_ds.labels)),
                 (torch.from_numpy(testdata), torch.from_numpy(test_ds.labels))    )

def download():
    (X_train, y_train), (X_test, y_test) = MNIST()
    print('MNIST:')
    print("Trainset:", X_train.shape, y_train.shape, np.unique(y_train))
    print("Testset: ", X_test.shape, y_test.shape, np.unique(y_test))
    (X_train, y_train), (X_test, y_test) = SVHN()
    print('SVHN:')
    print("Trainset:", X_train.shape, y_train.shape, np.unique(y_train))
    print("Testset: ", X_test.shape, y_test.shape, np.unique(y_test))

