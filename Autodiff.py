import numpy as np
class Variable(object):
    __variable_name = "Variable"
    __numero = 0
    def __init__(self, value, name, shape):
        self.name = name
        self.shape = shape
        self.value = None
        self.set_value(value)

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value