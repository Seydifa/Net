"""
Auteur : Moussa Seydi Faye
email : mseydifa@gmail.com
Profession : Ingenieur en mathematique et numerique
"""

import numpy as np

class activation(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented()

    def derivate(self):
        raise NotImplemented()

class Linear(activation):
    __name = "Linear"
    __numero = 0
    def __init__(self):
        super(Linear, self).__init__()
        self.name = Linear.__name + f" {Linear.__numero}"
        Linear.__numero += 1

    def __call__(self, x):
        return x

    def derivate(self, x):
        return np.ones_like(x)

class Sigmoid(activation):
    __name = "Sigmoid"
    __numero = 0
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = Sigmoid.__name
        self.numero = Sigmoid.__numero
        Sigmoid.__numero += 1

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivate(self,x):
        return self(x)*(1-self(x))

"""
<--- Tanh --->
"""
class Tanh(activation):
    __name = "Tanh"
    __numero = 0
    def __init__(self):
        super(Tanh, self).__init__()
        self.name = Tanh.__name
        self.__numero = Tanh.__numero
        Tanh.__numero += 1

    def __call__(self, x):
        return np.tanh(x)

    def derivate(self, x):
        return 1 - np.tanh(x)**2

    def __repr__(self):
        return self.name + f"_{self.__numero}"

class Relu(activation):
    __name = "Relu"
    __numero = 0
    def __init__(self):
        super(Relu, self).__init__()
        self.name = Relu.__name
        self.__numero = Relu.__numero
        Relu.__numero += 1

    def __call__(self, x):
        return np.where(x <= 0, 0, x)

    def derivate(self, x):
        return np.where(x <= 0, 0, 1)

    def __repr__(self):
        return self.name + f"_{self.__numero}"

class LeakyRelu(activation):
    __name = "LeakyRelu"
    __numero = 0
    def __init__(self,alpha=0.1):
        super(LeakyRelu, self).__init__()
        self.name = LeakyRelu.__name
        self.__numero = LeakyRelu.__numero
        LeakyRelu.__numero += 1
        self.alpha = 0.1

    def __call__(self, x):
        return np.where(x <= 0, self.alpha, x)

    def derivate(self, x):
        return np.where(x <= 0, 0, 1)

    def __repr__(self):
        return self.name + f"_{self.__numero}"

def get(name):
    activation = {"sigmoid": Sigmoid(), "tanh": Tanh(), "relu": Relu(), "leakyrelu": LeakyRelu(),
                  "linear": Linear()}
    return activation[name]

if __name__ == "__main__":
    print("ok work successfully")