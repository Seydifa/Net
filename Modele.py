"""
Auteur : Moussa Seydi Faye
email : mseydifa@gmail.com
Profession : Ingenieur en mathematique et numerique
"""

from .Couches import Couche
from .Optimizers import optimizer as Optimizer, get_optimizers
from .Losses import Loss, get_loss
from copy import deepcopy
import numpy as np
import time
from .Utile import Barreprogress


class Perceptron(object):
    __type = "Perceptron Multicouche"
    __numero = 0

    def __init__(self, input_dim: int, shape: list[int] | tuple[int] = None):
        """
        Constructeur de votre modele
        :param input_dim : La dimension des variables de votre modele
        :param shape : Les dimensions des donnees de votre modele
        """
        self.input_dim = input_dim
        self.input_shape = (None, self.input_dim) if shape is None else shape
        self.output_shape = (None, None)
        self.Couches = []
        self.built = False
        self.name = Perceptron.__type + f" {Perceptron.__numero}"
        self.loss = None
        self.history = {}

    def fit(self, x, y, epochs: int, batch_size: int = 1, optimizer: Optimizer | str = 'rmsprop',
            loss: Loss | str = 'crossentropy', verbose: bool = False, shuffle: bool = False,
            validation_split: float = 0):
        """
        :param x:
        :param y:
        :param epochs:
        :param batch_size:
        :param optimizer:
        :param loss:
        :param verbose:
        :param shuffle:
        :param validation_split:
        :return: history
        """
        optimizer = get_optimizers(optimizer) if isinstance(optimizer, str) else optimizer
        for couche in self.Couches:
            couche.optimizer = deepcopy(optimizer)
        self.loss = get_loss(loss) if isinstance(loss, str) else loss
        assert len(y.shape) > 1, f"y doit etre une matrice a deux dimension ! {y.shape}"
        self.history = {'loss': [], 'epochs': [], 'temps': []}
        validation = {'bool': False, 'loss': []}
        if validation_split:
            self.history['validation_loss'] = []
            train_indices, validation_indices = Perceptron.validation_split(test_size=validation_split,
                                                                            input_size=int(x.shape[0]),
                                                                            shuffle=shuffle)
        train_x = x[train_indices, ::] if validation_split else x
        train_y = y[train_indices, ::] if validation_split else y
        validation_x = x[validation_indices, ::] if validation_split else None
        validation_y = y[validation_indices, ::] if validation_split else None
        batch_num = train_x.shape[0] // batch_size
        batch_indices = Perceptron.batch(input_size=train_x.shape[0], batch_num=batch_num, shuffle=shuffle)
        for epoch in range(epochs):
            loss_history = []
            t1 = time.time()
            for j, ind in enumerate(batch_indices):
                x_train = train_x[ind, :]
                y_train = train_y[ind, :]
                self.backward(y_true=y_train, y_pred=self(x_train))
                self.gradient(x_train)
                for couche in self.Couches:
                    if couche.trainable:
                        if "dense" in couche.name.lower():
                            new_params = couche.optimizer.update_rule([couche.w, couche.b],
                                                                      [couche.dw, couche.db])
                        elif 'reccurent' in couche.name.lower():
                            new_params = couche.optimizer.update_rule(
                                [couche.cellule.w, couche.cellule.b, couche.cellule.sw, couche.cellule.sb],
                                [couche.cellule.dw, couche.cellule.db, couche.cellule.dsw, couche.cellule.dsb]
                            )

                        couche.update(*new_params)
                loss_history.append(self.loss(y_true=y_train, y_pred=self(x_train)))
                if verbose:
                    if validation_split:
                        validation['bool'] = True
                        validation['loss'] = self.loss(y_true=validation_y, y_pred=self(validation_x))
                    Barreprogress(epochs, epoch, loss_history[-1], j + 1, batch_num, validation)
            t2 = time.time()
            temps = t2 - t1
            self.history['loss'].append(np.mean(loss_history))
            self.history['epochs'].append(epoch + 1)
            self.history['temps'].append(temps)
            if validation_split:
                self.history['validation_loss'].append(self.loss(y_true=validation_y, y_pred=self(validation_x)))
        return self.history

    @staticmethod
    def batch(input_size: int, batch_num: int, shuffle: bool):
        indices = np.arange(input_size)
        if shuffle:
            np.random.shuffle(indices)

        folds = np.split(indices, batch_num)
        if len(folds) > batch_num:
            folds[batch_num] = np.stack([folds[batch_num], folds[batch_num + 1]])
            folds.pop(batch_num + 1)
        return folds

    @staticmethod
    def validation_split(test_size:float, input_size:int, shuffle: bool):
        """

        :param prop: La tailles des donnees de test en proportion
        :param input_shape: le nombre de ligne de nos donnees
        :param shuffle: Condiction de melange eleatoire des donnees
        :return:
        """
        indices = np.arange(input_size)
        if shuffle:
            np.random.shuffle(indices)
        ind = int(test_size * input_size)
        train_indices = indices[:-ind]
        validation_indices = indices[-ind:]
        return train_indices, validation_indices

    def build(self, input_dim: int):
        """
        Construit votre modele
        :param input_dim:
        :return:
        """
        for couche in self.Couches:
            if ~couche.built:
                couche.build(input_dim)
            input_dim = couche.output_dim
        self.built = True

    def ajout(self, couche: Couche | list[Couche]):
        """
        Ajoute une couche de neurones a votre modele
        :param couche: une couche de neurones
        """
        self.Couches.append(couche) if isinstance(couche, Couche) else self.Couches.extend(couche)
        self.build(input_dim=self.input_dim)
        self.output_shape = (None, couche.output_dim) if isinstance(couche, Couche) else (None, couche[-1].output_dim)
        self.built = True

    def __call__(self, x):
        """
        Appelle votre modele
        :param x: Les donnees d'entree
        :return: retourne le resultat une matrice de taille (nombre de donnees, nombre de neurones de la derniere couche)
        """
        assert self.Couches, "Veuillez ajouter des couches de neurones a votre modele !"
        assert self.built, "Veuillez construire votre modele !"
        a = x
        for couche in self.Couches:
            a = couche(a)
        return a

    def backward(self, y_true, y_pred):
        """
        Calcule l'erreur delta de chaque couche
        :param y_true:
        :param y_pred:
        :return:
        """
        error = self.loss.derivate(y_true, y_pred)
        for couche in reversed(self.Couches):
            error = couche.backward(error)
        return None

    def gradient(self, x):
        """
        Calcule les gradientes de votre modele
        :param x:
        :return:
        """
        a = x
        for couche in self.Couches:
            a = couche.gradient(a)
        return None

    def __repr__(self):
        return self.name


if __name__ == "__main__":
    print("ok work succesfully")
