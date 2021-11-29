import numpy as np
import copy


def minimise(problem,
             algorithm,
             termination=None,
             **kwargs):

    # Set up problem
    if algorithm.problem is None:
        algorithm.setup(problem, **kwargs)

    # Run optimisation
    algorithm.solve()


