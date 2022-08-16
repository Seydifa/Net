import numpy as np
from .Utile import Lr
class optimizer:
    def __init__(self, lr: float | Lr = 0.01):
        self.lr = lr
        self.history = {}
        self.params = None
        self.grads = None

    def update(self, params, grads):
        self.params = params
        self.grads = grads
        self.update_rule()

    def update_rule(self):
        raise NotImplementedError()


class SGD(optimizer):
    __type = 'SGD'
    __numero = 0

    def __init__(self, lr=0.01):
        super().__init__(lr=lr)
        self.name = SGD.__type + '_' + str(SGD.__numero)
        SGD.__numero += 1

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def update_rule(self, params, grads):
        new_params = []
        for param, grad in zip(params, grads):
            param = param - self.lr * grad
            new_params.append(param)
        return new_params


class RMSprop(optimizer):
    __type = 'RMSprop'
    __numero = 0

    def __init__(self, lr=0.01, decay_rate=0.99):
        super().__init__(lr=lr)
        self.name = RMSprop.__type + '_' + str(RMSprop.__numero)
        RMSprop.__numero += 1
        self.decay_rate = decay_rate
        self.cache = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def update_rule(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(param) for param in params]
        new_cache = []
        new_params = []
        for param, grad, cache in zip(params, grads, self.cache):
            cache = self.decay_rate * cache + (1 - self.decay_rate) * grad * grad
            param = param - self.lr * grad / (np.sqrt(cache) + 1e-7)
            new_params.append(param)
            new_cache.append(cache)
        self.cache = new_cache
        return new_params

class Adam(optimizer):
    __type = 'Adam'
    __numero = 0

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        super().__init__(lr=lr)
        self.name = Adam.__type + '_' + str(Adam.__numero)
        Adam.__numero += 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def update_rule(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]
        new_m = []
        new_v = []
        new_params = []
        for param, grad, m, v in zip(params, grads, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad * grad
            param = param - self.lr * m / (np.sqrt(v) + 1e-7)
            new_params.append(param)
            new_m.append(m)
            new_v.append(v)
        self.m = new_m
        self.v = new_v
        return new_params


def get_optimizers(name):
    optimizer_dict = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adam': Adam
    }
    return optimizer_dict[name]
