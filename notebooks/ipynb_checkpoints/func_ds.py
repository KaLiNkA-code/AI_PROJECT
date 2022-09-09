import numpy as np
from matplotlib import pyplot as plt
x_plot = []
y_plot = []


def calculate_model_output(w, b, x):
    m = x.shape  # the number of training examples
    f_wb = np.zeros(m)
    for i in range(len(x)):
        f_wb[i] = w * x[i] + b
    return f_wb


# MSE loss function
def mse_loss(val_pred, val_true):
    squared_error = (val_pred - val_true) ** 2
    sum_squared_error = np.sum(squared_error)
    loss = sum_squared_error / val_true.size
    return loss


def train_one_step(X, w, y_train, N, LR, i):
    global x_plot, y_plot
    f = X.dot(w)
    err = f - y_train
    grad = 2 * X.T.dot(err) / N
    w -= LR * grad
    if i % 10 == 0:
        x_plot.append(i)
        y_plot.append(err.mean())
        return err.mean
