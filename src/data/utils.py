import torch
import torch.nn.functional as F

from . import datasets

def ensure_tensors(*maybe_tensors):
    return tuple(x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in maybe_tensors)

def sequential_lables(labels, order):
    labels, order = ensure_tensors(labels, order)
    # convert targets selection to labels without gaps, and in the same order as in use_targets
    _, label_indices = labels.unique(return_inverse=True)
    _, order_indices = order.unique(return_inverse=True)
    new_classes = order_indices.sort().indices
    return new_classes[label_indices]

def subproblem(data, targets, classes):
    data, targets, classes = ensure_tensors(data, targets, classes)
    targets_oh = F.one_hot(targets).bool()
    use_targets_oh = F.one_hot(classes, num_classes=targets_oh.shape[1]).any(0).bool()
    # mask with rows having one of the used targets
    selection = (targets_oh & use_targets_oh).any(1)
    return data[selection], targets[selection]

# def set_tuples2subproblem(classes, *Xy_tuples):
#     return tuple(subproblem(X, y, classes) for X, y in Xy_tuples)

def load_dataset(dataset_name='MNIST', classes=None, new_sequential_classes=False, standardize=True, asnumpy=True):
    (X_train, y_train), (X_test, y_test) = getattr(datasets, dataset_name)(asnumpy=False)
    if classes:
        X_train, y_train = subproblem(X_train, y_train, classes)
        X_test, y_test = subproblem(X_test, y_test, classes)
    else:
        classes = y_train.unique()
    if new_sequential_classes:
        y_train = sequential_lables(y_train, classes)
        y_test = sequential_lables(y_test, classes)
    if standardize:
        Xstd, Xmean = torch.std_mean(X_train.float(), dim=[0,2,3], keepdim=True)
        X_train = (X_train - Xmean)/Xstd
        X_test = (X_test - Xmean)/Xstd
    if asnumpy:
        X_train = X_train.numpy()
        y_train = y_train.numpy()
        X_test = X_test.numpy()
        y_test = y_test.numpy()
    return (X_train, y_train), (X_test, y_test)
