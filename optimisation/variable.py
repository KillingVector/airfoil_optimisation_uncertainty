import numpy as np

inf = 1e20


class Variable(object):

    def __init__(self, name, type, value, lower, upper, scale=1.0, choices=None, additional_values=None):

        self.name = name
        self.type = type
        self.scale = scale
        self.choices = None
        if self.type == 'c':
            if lower is None:
                self.lower = -inf
            else:
                self.lower = lower*scale
            if upper is None:
                self.upper = inf
            else:
                self.upper = upper*scale
            self.value = value*scale
            if additional_values is None:
                self.additional_values = additional_values
            else:
                self.additional_values = np.asarray(additional_values)*scale
        elif self.type == 'i':
            self.lower = lower
            self.upper = upper
            self.value = int(value)
            if additional_values is None:
                self.additional_values = additional_values
            else:
                self.additional_values = np.asarray(additional_values).astype(int)
        elif self.type == 'd':
            if choices is None:
                raise Exception('A discrete variable requires an array of choices')
            self.choices = choices
            self.lower = 0
            self.upper = len(choices) - 1
            self.value = int(value)
            if additional_values is None:
                self.additional_values = additional_values
            else:
                self.additional_values = np.asarray(additional_values).astype(int)

