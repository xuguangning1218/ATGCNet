import numpy as np

def MAE(target, preds, masker):
    return np.mean(np.abs(target[masker] - preds[masker]))

def MSE(target, preds, masker):
    return np.mean(np.square(target[masker] - preds[masker]))