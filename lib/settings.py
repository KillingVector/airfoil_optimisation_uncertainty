import numpy as np


class Settings(object):
    def __init__(self, n_core=1, mesher=None, solver=None, use_python_xfoil=False, plot=False, uncertainty = False, uncertainty_tag = ['v','T']):

        # CFD settings
        self.n_core = n_core
        self.mesher = mesher
        self.solver = solver
        self.use_python_xfoil = use_python_xfoil
        self.uncertainty = Uncertainty()

        # Graphics settings
        self.plot = plot
        
        
class Uncertainty(object):
    def __init__(self):
        
        self.run    = False # default
        self.indep  = True  # default - assume independent
        self.tag    = 'vT'
        self.sigma  = 1
        self.order  = 2
        self.initial= 'normal'
        self.rule   = 'Gaussian'
        
        self.dist   = 0.1 # fraction of variable, single val or list len == #var
    
    def trigger(self):
        self.run    = True
        
    def set(self, tag = 'vt', sigma = 1, order = 2, initial = 'normal', rule = 'Gaussian', dist = 0.1):
        import warnings
        if tag.lower() in ['vt','tv','v t', 't v']:
            self.tag = 'vT'
        elif tag.lower() in ['rm','mr','machre','remach','mre','rem']:
            self.tag = 'MRe'
        else:
            warnings.warn('WARNING: Unsupported uncertainty tag: using default velocity, Temperature')
            # else use default vT
        self.order = order
        if initial.lower() in ['norm','normal','gauss','gaussian']:
            self.initial = 'normal'
        elif initial.lower() in ['uniform','single','average']:
            self.initial = 'uniform'
        else:
            warnings.warn('WARNING: Unsupported uncertainty distribution: using default normal initial distribution')
            # else use default normal
        self.rule   = rule # has chaospy warnings
        if not type(dist) == type(list()):
            if dist >= 1:
                warnings.warn('WARNING: Incorrect uncertainty dist value: dist should be fraction/percentage (<1), converting...')
                dist    = dist / 100
        else:
            for i,d in enumerate(dist):
                if d >=1 :
                    dist[i] = d/100
        self.dist   = dist

        if sigma < 0:
            warnings.warn('WARNING: Incorrect sigma: standard deviation cannot be negative, setting to positive.')
            sigma   = np.abs(sigma)
        self.sigma  = sigma
        
            
            
