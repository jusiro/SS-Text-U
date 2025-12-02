import torch

from .acc import accuracy
from .aca import average_class_wise_accuracy

def evaluate(output, target):

    # Overall accuracy
    acc = accuracy(torch.tensor(output).float(), torch.tensor(target).long())[0].item()

    # Balanced accuracy
    aca = average_class_wise_accuracy(output, target).item()

    return acc, aca
