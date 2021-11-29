import numpy as np


class Airfoil(object):

    def __init__(self, npt):

        self.npt = npt

        self.x = np.zeros(self.npt)
        self.z = np.zeros(self.npt)

        self.x_upper = np.zeros(int(np.floor(self.npt/2))+1)
        self.x_lower = np.zeros(int(np.floor(self.npt/2))+1)
        self.z_upper = np.zeros(int(np.floor(self.npt/2))+1)
        self.z_lower = np.zeros(int(np.floor(self.npt/2))+1)

        self.delta_z_te = 0.0

    def generate_section(self, **kwargs):
        self._generate_section(**kwargs)

    def _generate_section(self, **kwargs):
        pass

