import numpy as np


from optimisation.model.surrogate import Surrogate


class EnsembleSurrogate(Surrogate):
    def __init__(self, n_dim, l_b, u_b, models=None, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # Todo: Implement surrogates for each model in models (models is a list of models)
        # self.models = SurrogateHandler(models)

