from lib import config, util
from lib.design import Design
from lib.settings import Settings
from lib.utils import setup_solver

from cases.single_element_setup import SingleElementSetup
from optimisation.model.problem import Problem


def initialise(case_name, parametrisation_method, nr_design_variables):

    # Case name
    config.case_name = case_name

    # Clean up quick results files
    util.cleanup_quick_results()

    # Set up solver
    solver_name = 'mses'  # options are 'xfoil','xfoil_python', 'mses',su2','openfoam','su2_gmsh','openfoam_gmsh'
    setup_solver(solver_name, config)

    # # Design class instantiation
    design_objectives = ['max_weighted_lift_to_drag', 'max_thickness']
    # create an instance of the design class - if you want to run a new case you should set up a new application_id in
    # design.py (in the lib folder). design.py is where you set all the operating condtions, ...
    config.design = Design(parametrisation_method, nr_design_variables,
                           application_id=config.case_name, design_objectives=design_objectives)

    # Problem setup - this is where you add all the objectives and constraints to the setup (in lib)
    setup = SingleElementSetup()

    # Initialising optimisation problem (instance of the problem class which is generically set up in
    # optimisation/model/problem.py
    opt_prob = Problem(config.case_name, setup.obj_func, map_internally=True)

    # Set variables, constraints & objectives = call the case setup and add everything
    setup.do(opt_prob)
    config.design.n_con = opt_prob.n_con

    return setup, opt_prob


def run_obj_func(setup, x_dict, idx=0, **kwargs):

    # Run analysis using passed design variable vector
    obj, cons, performance = setup.obj_func(x_dict,idx, **kwargs)

    return obj, cons, performance

def initialise_single_objective(case_name, parametrisation_method, nr_design_variables):

    # Case name
    config.case_name = case_name

    # Clean up quick results files
    util.cleanup_quick_results()

    # Set up solver
    solver_name = 'mses'  # options are 'xfoil','xfoil_python', 'mses',su2','openfoam','su2_gmsh','openfoam_gmsh'
    setup_solver(solver_name, config)

    design_objectives = ['max_weighted_lift_to_drag']
    # create an instance of the design class - if you want to run a new case you should set up a new application_id in
    # design.py (in the lib folder). design.py is where you set all the operating condtions, ...
    config.design = Design(parametrisation_method, nr_design_variables,
                           application_id=config.case_name, design_objectives=design_objectives)

    # Problem setup - this is where you add all the objectives and constraints to the setup (in lib)
    setup = SingleElementSetup()

    # Initialising optimisation problem (instance of the problem class which is generically set up in
    # optimisation/model/problem.py
    # TODO this should be set back to map_internally = True
    opt_prob = Problem(config.case_name, setup.obj_func, map_internally=True)

    # Set variables, constraints & objectives = call the case setup and add everything
    setup.do(opt_prob)
    config.design.n_con = opt_prob.n_con

    return setup, opt_prob


def run_obj_func_single_objective(setup, x_dict, **kwargs):

    # Run analysis using passed design variable vector
    obj, cons, performance = setup.obj_func(x_dict,0, **kwargs)

    return obj, cons, performance
