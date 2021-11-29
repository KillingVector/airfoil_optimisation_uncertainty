import sys
import time
import os.path
import pickle, pickle5
import subprocess
import numpy as np
from mpi4py import MPI

from lib import config, util
from lib.design import Design

from cases.single_element_setup_sweep import SingleElementSetup
from optimisation.model.problem import Problem
from optimisation.algorithms.shamode_wo import SHAMODE
from optimisation.optimise import minimise

from optimisation.operators.sampling.seed_sampling import SeedSampling
from lib.utils import set_coefficient_bounds, generate_variables_from_xfoilrun, airfoil_analysis_xfoil
from lib.utils import normalise_seed_airfoils
from lib.utils import setup_solver_sweep as setup_solver
from lib.xfoil_seed_sampling import initialise_airfoils_with_xfoil
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival


def main(attempt_number):
    t = time.time()

    population_size = 15
    # ratio of final population to initial population
    population_ratio = 0.25
    nr_of_generations = 150

    # Case name
    # config.case_name = 'airfoil_testing'
    config.case_name = 'prop_0775r'
    # config.case_name = 'prop_4b_2pos_995rad_09pct'
    # config.case_name = 'prop_4b_2pos_235rad_21pct'
    # config.case_name = 'prop_4b_2pos_505rad_15pct'

    initialise_with_xfoil = False  # run xfoil on first generation to work out a reasonable starting angle for SU2
    # should only be used if a first optimisation was run with xfoil

    # Clean up quick results files
    util.cleanup_quick_results()

    # Set up solver
    solver_name = 'xfoil'  # options are 'xfoil','xfoil_python', 'mses',su2','openfoam','su2_gmsh','openfoam_gmsh'
    setup_solver(solver_name, config)

    # Design class instantiation
    parametrisation_method = 'CST'
    cst_order = 5
    nr_design_variables = 2 * (cst_order + 1)

    # parametrisation_method = 'CSTmod'
    # cst_order = 5
    # nr_design_variables = 2*(cst_order+1)+1

    # parametrisation_method = 'Bspline'
    # order = 10
    # nr_design_variables = 2 * order

    # parametrisation_method = 'Bezier'
    # order = 20
    # nr_design_variables = order

    # parametrisation_method = 'hickshenne'
    # order = 5
    # nr_design_variables = 2 * order

    # parametrisation_method = 'parsec'
    # nr_design_variables = 11

    design_objectives = ['max_weighted_lift_to_drag', 'max_lift']
    # design_objectives = ['max_weighted_lift_to_drag']
    # create an instance of the design class - if you want to run a new case you should set up a new application_id in
    # design.py (in the lib folder). design.py is where you set all the operating condtions, ...
    config.design = Design(parametrisation_method, nr_design_variables,
                           application_id=config.case_name, design_objectives=design_objectives)

    # Problem setup - this is where you add all the objectives and constraints to the setup (in lib)
    setup = SingleElementSetup()

    # Initialising optimisation problem (instance of the problem class which is generically set up in
    # optimisation/model/problem.py
    # TODO this should be set back to map_internally = True
    # opt_prob = Problem(config.case_name, setup.obj_func, map_internally=True, n_processors=16)
    opt_prob = Problem(config.case_name, setup.obj_func, map_internally=False, n_processors=1)

    # Set variables, constraints & objectives = call the case setup and add everything
    setup.do(opt_prob)
    config.design.n_con = opt_prob.n_con

    # Todo: This really shouldn't be here... (should be in sampling)
    sampling_method = SeedSampling()
    if initialise_with_xfoil:
        initialise_airfoils_with_xfoil(config.case_name, opt_prob, config.design)
    else:
        normalise_seed_airfoils(config.design)
    sampling_method = SeedSampling()

    # no seed sampling
    # algorithm = NSMTLBO(n_population=5 * opt_prob.n_var,
    #                      max_gen=100,
    #                      survival=TwoRankingSurvival(filter_infeasible=False))

    algorithm = SHAMODE(n_population=population_size* opt_prob.n_var,
                        population_ratio=population_ratio,
                        max_gen=nr_of_generations,
                        sampling=sampling_method,
                        survival=TwoRankingSurvival(filter_infeasible=False))

    # Flag for setting random seed for optimisation
    seed = 1

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

    # Set up copying of the files for Skawinski case
    savefilename = config.case_name + '_' + \
                   parametrisation_method + '_' + str(nr_design_variables) + '_' + \
                   'attempt' + '_' + str(attempt_number)
    with open('./results/optimisation_history_' + config.case_name + '.pkl', 'rb') as f:
        try:
            history = pickle.load(f)
        except ValueError:
            history = pickle5.load(f)
        f.close()

    with open('./results/optimisation_history_' + savefilename + '.pkl', 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    total_attempts = 2#5
    for counter in range(total_attempts):
        print('**************************************************************')
        print('           starting run number', counter)
        print('**************************************************************')
        main(attempt_number=counter)
    sys.exit()
