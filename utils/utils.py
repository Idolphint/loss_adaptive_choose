import numpy as np

def linear_gray_transpose(X):
    X = np.array(X)
    minn = np.min(X)
    maxx = np.max(X)
    k = 1 / (maxx - minn)
    b = -1 * minn / (maxx - minn)
    X = k * X + b
    return X.astype(np.float32)
