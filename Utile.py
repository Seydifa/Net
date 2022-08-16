import matplotlib.pyplot as plt
import numpy as np


class Accuracy:
    def __init__(self, seuil=0.5, name='accuracy', Class=[0, 1], labels=None):
        self.seuil = seuil
        self.name = name
        self.labels = None
        self.Class = Class

    def __call__(self, y_pred, y_true):
        y_pred = np.where(y_pred <= self.seuil, 0, 1)
        return np.mean(y_pred == y_true)


class Precision:
    def __init__(self, seuil=0.5, name='precision'):
        self.seuil = seuil
        self.name = name

    def __call__(self, y_pred, y_true):
        y_pred = np.where(y_pred <= self.seuil, 0, 1)
        return np.mean(y_pred == y_true)


import sys


def Barreprogress(total, progress, loss, batch, nb_batch, validation={}):
    barLength, status, k = 50, "", progress + 1
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.1f}% {} Loss = {:<10f} epoch = {}/{} batche = {}/{} ".format(
        "=" * block + "-" * (barLength - block), round(progress * 100, 0), status, loss, k, total, batch, nb_batch)
    if validation['bool']:
        text += f"validation Loss = {validation['loss']: <.6f}"
    sys.stdout.write(text)
    sys.stdout.flush()


def Plot_boundary(modele, x, y, figsize=(10, 10), ax=None):
    min_ = x.min(axis=0)
    max_ = x.max(axis=0)
    xx, yy = np.meshgrid(np.linspace(min_[0] - 0.5, max_[0] + 0.5, 50),
                         np.linspace(min_[1] - 0.5, max_[1] + 0.5, 50))

    x_ = np.c_[xx.ravel(), yy.ravel()]
    zz = modele(x_) if y.shape[1] == 1 else np.argmax(modele(x_), axis=1)
    zz = zz.reshape(xx.shape)
    y = y if y.shape[1] == 1 else np.argmax(y, axis=1)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    ax.contourf(xx, yy, zz, alpha=0.3)
    ax.scatter(x[:, 0], x[:, 1], c=y.ravel())
    return None

class Lr(object):
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, epoch):
        raise NotImplementedError()

class LearningRate(Lr):
    def __init__(self, lr: float = 0.01):
        super(LearningRate, self).__init__()
        self.lr = lr
        self.history = {}

    def __call__(self, step: int, loss: float | None = None):
        self.history[step] = step
        self.history['lr'] = self.lr
        if loss is not None:
            self.history[loss] = loss
        return self.lr

class AdaptativeLr(Lr):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=1e-2,
                 step_size=2000, factor=0.1, min_lr=1e-4,
                 patience=10, verbose=True):
        super(AdaptativeLr, self).__init__()
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.count = 0
        self.best_loss = np.inf
        self.lr = self.base_lr

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1

        if self.count > self.patience:
            self.lr = max(self.lr * self.factor, self.min_lr)
            self.count = 0
            if self.verbose:
                print('Reducing learning rate to: {}'.format(self.lr))
        return self.lr