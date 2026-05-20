import torch

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean()