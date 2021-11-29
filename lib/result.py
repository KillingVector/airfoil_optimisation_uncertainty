import numpy as np


class Result(object):
    def __init__(self, npt):

        if npt == 1:
            self.c_l = -1.0
            self.c_d = 1.0
            self.c_m = 1.0
            self.iter = 0
        else:
            self.c_l = -1.0*np.ones(npt)
            self.c_d = np.ones(npt)
            self.c_m = np.ones(npt)
            self.iter = np.zeros(npt, dtype=int)


class Result2(object):
    def __init__(self, npt):

        if npt == 1:
            self.c_l = -1.0
            self.c_d = 1.0
            self.c_m = 1.0
            self.iter = 0
            self.alpha = 0
        else:
            self.c_l = -1.0*np.ones(npt)
            self.c_d = np.ones(npt)
            self.c_m = np.ones(npt)
            self.iter = np.zeros(npt, dtype=int)
            self.alpha = np.zeros(npt)


class SU2Results(object):
    def __init__(self):
        self.cl = 0.0
        self.cd = 0.0
        self.cm = 0.0
        self.iter = 0.0
        self.residual = 0.0