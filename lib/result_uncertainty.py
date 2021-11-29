import numpy as np

# l/d included to explicitly generate statistics
class Result(object):
    def __init__(self, npt):

        if npt == 1:
            self.c_l = -1.0
            self.c_d = 1.0
            self.c_m = 1.0
            self.l_d = -1.0 # lift on drag
            self.iter = 0
            self.cl_stats = -1*np.ones(4) # std, var, skew, kurt
            self.cl_sens = -1*np.ones(2) # sobol sensitivities
            self.cd_stats = -1*np.ones(4) # std, var, skew, kurt
            self.cd_sens = -1*np.ones(2) # sobol sensitivities
            self.cm_stats = -1*np.ones(4) # std, var, skew, kurt
            self.cm_sens = -1*np.ones(2) # sobol sensitivities
            self.ld_stats = -1*np.ones(4) # std, var, skew, kurt
            self.ld_sens = -1*np.ones(2) # sobol sensitivities
        else:
            self.c_l = -1.0*np.ones(npt)
            self.c_d = np.ones(npt)
            self.c_m = np.ones(npt)
            self.l_d = np.ones(npt)
            self.iter = np.zeros(npt, dtype=int)
            self.cl_stats = -1 * np.ones((npt,4)) # std, var, skew, kurt
            self.cl_sens = -1 * np.ones((npt,2)) # sobol sensitivities
            self.cd_stats = -1 * np.ones((npt,4)) # stats
            self.cd_sens = -1 * np.ones((npt,2)) # sobol sens
            self.cm_stats = -1 * np.ones((npt,4)) # stats
            self.cm_sens = -1 * np.ones((npt,2)) # sobol sens
            self.ld_stats = -1 * np.ones((npt,4)) # stats
            self.ld_sens = -1 * np.ones((npt,2)) # sobol sens
            
            




