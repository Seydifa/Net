import numpy as np
global epsilon
epsilon = 1e-9
"""
epsilon = 1e-7 : Pour le stabilite numerique
"""
class Loss(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented()

    def derivate(self, *args, **kwargs):
        raise NotImplemented()

class CrossEntropy(Loss):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def __call__(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + epsilon))

    def derivate(self, y_true, y_pred):
        return -y_true / (y_pred + epsilon)

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def __call__(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivate(self, y_true, y_pred):
        return -(y_true-y_pred) / (y_pred - y_pred**2 + epsilon)

class MSE(Loss):
    def __init__(self):
        super(MSE, self).__init__()

    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def derivate(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_pred.shape[0]

class AbsoluteError(Loss):
    def __init__(self):
        super(AbsoluteError, self).__init__()

    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def derivate(self, y_true, y_pred):
        return -1 * (y_true - y_pred) / y_pred.shape[0]

class MeanSquaredError(Loss):
    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def derivate(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_pred.shape[0]

class RootMeanSquaredError(Loss):
    def __init__(self):
        super(RootMeanSquaredError, self).__init__()

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def derivate(self, y_true, y_pred, epsilon=1e-7):
        return -(2 * (y_true - y_pred) / y_pred.shape[0])/(self(y_true, y_pred) + epsilon)

def get_loss(name):
    Lossdict = {"crossentropy": CrossEntropy,
                "binarycrossentropy": BinaryCrossEntropy,
                "mse":MSE, "absoluteerror": AbsoluteError,
                "rootsquarederror": RootMeanSquaredError}
    return Lossdict[name]()

if __name__ == "__main__":
    print("ok work successfull")