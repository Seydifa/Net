"""
Auteur : Moussa Seydi Faye
email : mseydifa@gmail.com
Profession : Ingenieur en mathematique et numerique
"""

from .Activations import activation as Activation, get as get_activation
from numpy import random
import numpy as np


class Couche(object):
    """
    class : Couche
    """
    __numero = 0

    def __init__(self, name):
        """
        Constructeur de votre modele
        :param name:
        """
        self.__name = name
        self.__numero = Couche.__numero
        Couche.__numero += 1
        self.built = False
        self.output_shape = None
        self.input_shape = None
        self.delta = None
        self.dw = None
        self.db = None
        self.name = self.__name

    def build(self, output_shape):
        raise NotImplemented()

    def __str__(self):
        return self.__name + f"_{self.__numero}"

    def __repr__(self):
        return self.__name + f"_{self.__numero}"

    def __call__(self, *args, **kwargs):
        raise NotImplemented()


class Dropout(Couche):
    __type = "Dropout"
    __numero = 0

    def __init__(self, rate: float = 0.2):
        self.name = Dropout.__type + f" {Dropout.__numero}"
        Dropout.__numero += 1
        super(Dropout, self).__init__(self.name)
        self.rate = rate
        self.built = False
        self.trainable = False

    def build(self, input_dim):
        self.output_dim = input_dim
        self.input_dim = input_dim
        self.built = True

    def __call__(self, x):
        self.a = x * np.random.binomial(1, 1 - self.rate, x.shape)
        return self.a

    def backward(self, error):
        return error

    def gradient(self, a):
        return self.a


class Rescale(Couche):
    __type = "Rescale"
    __numero = 0

    def __init__(self, scale=1):
        self.__type = Rescale.__type
        self.__numero = Rescale.__numero
        self.name = self.__type + f" {self.__numero}"
        super(Rescale, self).__init__(self.name)
        Rescale.__numero += 1
        self.trainable = False
        self.built = False
        self.output_shape = None
        self.input_shape = None
        self.scale = scale
        self.a = None

    def build(self, input_shape):
        self.output_dim = input_shape
        self.input_dim = input_shape
        self.built = True

    def __call__(self, x):
        self.a = self.scale * x
        return self.a

    def backward(self, error):
        return self.a * error

    def gradient(self, a):
        return self.a


class Normalisation(Couche):
    __type = "Normalisation"
    __numero = 0

    def __init__(self):
        self.__type = Normalisation.__type
        self.__numero = Normalisation.__numero
        self.name = self.__type + f" {self.__numero}"
        super(Normalisation, self).__init__(self.name)
        Normalisation.__numero += 1
        self.trainable = False
        self.built = False
        self.output_shape = None
        self.input_shape = None
        self.mean = None
        self.std = None
        self.a = None

    def build(self, input_shape):
        self.output_dim = input_shape
        self.input_dim = input_shape
        self.built = True

    def __call__(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.a = (x - self.mean) / (self.std + 1e-8)
        return self.a

    def backward(self, error):
        return error / (self.std + 1e-8)

    def gradient(self, a):
        return self.a


class Dense(Couche):
    __name = "Dense"

    def __init__(self, nb_neurone: int, activation: Activation | str = 'sigmoid',
                 shape: list[int] | tuple[int] | None = None,
                 biais=True):
        super(Dense, self).__init__(Dense.__name)
        self.unit = nb_neurone
        self.activation = activation if isinstance(activation, Activation) else get_activation(activation)
        self.input_shape = None if shape is None else shape
        self.output_shape = None if shape is None else (shape[-1], nb_neurone)
        self.input_dim = None if shape is None else self.input_shape[-1]
        self.output_dim = nb_neurone
        self.biais = biais
        self.trainable = True
        self.variables = []
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = random.randn(self.unit, input_shape)/np.sqrt(input_shape)
        self.b = random.randn(1, self.unit) if self.biais else np.zeros(shape=(1, self.unit))
        self.variables = [self.w, self.b]
        self.built = True

    def __call__(self, x, ):
        assert self.built, "Veuillez contruire votre couches d'abord !"
        self.z = x.dot(self.w.T) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, error):
        self.delta = error * self.activation.derivate(self.z)
        return self.delta @ self.w

    def gradient(self, a):
        self.dw = self.delta.T.dot(a)
        self.db = self.delta.sum(axis=0, keepdims=True)
        return self.a

    def update(self, w, b):
        self.w = w
        self.b = b
        return

    def optimizer(self):
        raise NotImplemented()


class Cellule(object):
    __numero = 0

    def __init__(self, nb_neurone: int,
                activation: Activation | str,
                reccurent_activation: Activation | str, biais=True):
        self.nb_neurone = nb_neurone
        self.biais = biais
        self.activation = activation if isinstance(activation, Activation) else get_activation(activation)
        self.reccurent_activation = reccurent_activation if isinstance(reccurent_activation, Activation)\
            else get_activation(reccurent_activation)
        self.input_dim = None
        self.w = None
        self.sw = None
        self.b = None
        self.sb = None
        self.built = False
        self.sz = None
        self.sa = None
        self.iz = None
        self.ia = None
        self.dw = None
        self.dsw = None
        self.db = None
        self.dsb = None

    def build(self, input_dim, state_dim):
        self.w = random.randn(self.nb_neurone, input_dim)
        self.sw = random.randn(self.nb_neurone, state_dim)
        self.b = random.randn(1, self.nb_neurone)
        self.sb = random.randn(1, self.nb_neurone)
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.built = True

    def __call__(self, x, s):
        self.iz = x.dot(self.w.T) + self.b
        self.ia = self.activation(self.iz)

        self.sz = s.dot(self.sw.T) + self.sb
        self.sa = self.reccurent_activation(self.sz)

        return self.sa + self.ia, self.ia

    def backward(self, error):
        self.delta = error*(self.activation.derivate(self.iz))
        self.deltas = error*(self.reccurent_activation(self.sz))
        return self.delta @ self.w


# <------------------- RNN ------------------------->
class RNN(Couche):
    __name = "Reccurent_neural_network"

    def __init__(self, nb_neurone: int, activation: Activation | str, reccurent_activation: Activation | str,
                 shape: list[int] | tuple[int] | None = None,
                 biais=True, recurrent_bias=False):
        self.cellule = Cellule(
            nb_neurone=nb_neurone,
            activation=activation,
            reccurent_activation=reccurent_activation,
            biais=biais
        )
        super(RNN, self).__init__(name=RNN.__name)
        self.input_shape = None if shape is None else shape
        self.output_shape = None if shape is None else (shape[-1], nb_neurone)
        self.input_dim = None if shape is None else self.input_shape[-1]
        self.nb_neurone = nb_neurone
        self.output_dim = nb_neurone
        self.biais = biais
        self.trainable = True
        self.delta = None
        self.deltas = None
        self.built = False
        self.state = None


    def build(self, input_dim, state_dim=None):
        state_dim = self.nb_neurone if state_dim is None else state_dim
        self.cellule.build(input_dim, state_dim)
        self.built = True

    def __call__(self, x, init_state=None):
        assert self.built, "Veuillez construire votre modele !"
        s = np.zeros((1, self.cellule.state_dim)) if init_state is None else init_state
        self.state = [s]
        self.a, s = self.cellule(x[0, ::], s)
        for v in x[1:, ::]:
            y, s = self.cellule(v, s)
            self.a = np.vstack([self.a, y])
            self.state.append(s)
        self.state = np.vstack(self.state)
        return self.a

    def backward(self, error):
        errorback = []
        self.delta = []
        self.deltas = []
        for err in reversed(error):
            errorback.append(self.cellule.backward(err))
            self.delta.append(self.cellule.delta)
            self.deltas.append(self.cellule.deltas)
        self.delta = np.vstack(self.delta)
        self.deltas = np.vstack(self.deltas)
        return np.vstack(errorback)

    def gradient(self, a):
        self.cellule.dw = self.delta.T.dot(a)
        self.cellule.db = self.delta.sum(axis=0, keepdims=True)
        self.cellule.dsw = self.deltas.T.dot(self.state)
        self.cellule.dsb = self.deltas.sum(axis=0, keepdims=True)
        return self.a

    def update(self, w, b, sw, sb):
        self.cellule.w = w
        self.cellule.b = b
        self.cellule.sw = sw
        self.cellule.sb = sb
        return None

    def optimizer(self):
        raise NotImplemented()