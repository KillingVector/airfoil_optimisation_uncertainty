import sys
import time
from mpi4py import MPI

from lib import config, util
from lib.design import Design
from lib.settings import Settings

from cases.single_element_setup import SingleElementSetup
from optimisation.model.problem import Problem
from optimisation.algorithms.nsga2 import NSGA2
from optimisation.optimise import minimise

from optimisation.operators.sampling.seed_sampling import SeedSampling
from lib.utils import normalise_seed_airfoils

from optimisation.surrogate.surrogate_strategy import SurrogateStrategy
from optimisation.surrogate.models.rbf import RadialBasisFunctions


def main():

    t = time.time()

    # Case name
    config.case_name = 'airfoil_testing'

    # Clean up quick results files
    util.cleanup_quick_results()

    # Settings
    # config.settings = Settings(n_core=1, mesher='gmsh', solver='openfoam')
    config.settings = Settings(n_core=1, mesher=None, solver='xfoil')
    # config.settings = Settings(n_core=32, mesher=None, solver='mses')

    # Design class instantiation
    parametrisation_method = 'CST'
    cst_order = 5
    nr_design_variables = 2*(cst_order+1)
    design_objectives = ['max_lift_to_drag', 'max_lift_to_drag']
    config.design = Design(parametrisation_method, nr_design_variables,
                           application_id='propeller_opt_3b_fixed_0775r', design_objectives=design_objectives)

    # Problem setup
    setup = SingleElementSetup()

    # Initialising optimisation problem
    opt_prob = Problem(config.case_name, setup.obj_func, map_internally=False)

    # Set variables, constraints & objectives
    setup.do(opt_prob)
    config.design.n_con = opt_prob.n_con

    # Add specific functions to problem to allow calculation paired with surrogate calculations
    opt_prob.add_specific_funcs(setup)

    # Surrogate strategy instance
    surrogate = SurrogateStrategy(opt_prob,
                                  obj_surrogates=[RadialBasisFunctions(opt_prob.n_var,
                                                                       l_b=opt_prob.x_lower, u_b=opt_prob.x_upper,
                                                                       c=0.5, p_type='linear') for _ in range(opt_prob.n_obj)],
                                  # cons_surrogates=None,
                                  cons_surrogates=[RadialBasisFunctions(opt_prob.n_var,
                                                                        l_b=opt_prob.x_lower, u_b=opt_prob.x_upper,
                                                                        c=0.5, p_type='linear') for _ in range(opt_prob.n_con)],
                                  n_training_pts=5*opt_prob.n_var, n_infill=5, max_real_f_evals=1000,
                                  opt_npop=5*opt_prob.n_var, opt_ngen=25,
                                  plot=False, print=True)

    # Todo: This really shouldn't be here... (should be in sampling)
    sampling_method = SeedSampling()
    normalise_seed_airfoils(config.design)

    algorithm = NSGA2(n_population=4*opt_prob.n_var,
                      max_gen=100,
                      sampling=sampling_method,
                      surrogate=surrogate)

    # Flag for setting random seed for optimisation
    seed = None

    # Hot-start flag
    hot_start = False

    # Flag for passing initial values of design variables to population
    x_init = True

    # Run optimisation
    minimise(opt_prob,
             algorithm,
             seed=seed,
             hot_start=hot_start,
             x_init=x_init,
             save_history=True)

    MPI.COMM_WORLD.barrier()

    elapsed = time.time() - t
    print('Elapsed time: {0:f} seconds'.format(elapsed))

    sys.exit()


if __name__ == '__main__':

    main()


