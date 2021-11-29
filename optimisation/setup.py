import numpy as np


class Setup:

    def __init__(self):

        self.null = 0

    def do(self, prob, **kwargs):

        self.set_variables(prob, **kwargs)
        self.set_constraints(prob, **kwargs)
        self.set_objectives(prob, **kwargs)

        prob.finalise()

    def set_variables(self, prob, **kwargs):
        pass

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        pass

    def obj_func(self, x_dict, **kwargs):
        pass



