import numpy as np
import os, copy
import subprocess
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from lib import config
from lib.geometry import rotate_element
from lib.graphics import plot_geometry
from lib.result import Result
from lib.cfd_lib import gmsh_util, construct2d_util, xfoil_util, mses_util
from lib.util import write_obj_cons
from lib import flight

def airfoil_analysis(design, sol_idx, write_quick_history=True, plot=False, print_output=False, debug=False, **kwargs):
    """
    sets up the analysis of the airfoil
    :param design: design class instance that contains the airfoil object
    :param sol_idx: identifier for the solution to prevent other cores to overwrite the solution of the current core
    :param write_quick_history: boolean to write the solution in the quick history files or not
    :param plot: boolean to plot the geometry or not
    :param kwargs: other arguments
    :return:
    """

    # create the airfoil instance

    airfoil = design.airfoil
    delta_z_te = 0.0025

    # Generate the section and write airfoil to file
    airfoil.generate_section(design.shape_variables, design.n_pts, delta_z_te=delta_z_te)
    front_element_unscaled = copy.deepcopy(airfoil)

    # scale main airfoil
    airfoil.scale_section(scale_factor=design.scale_element1)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists('./airfoils/'):
        os.makedirs('./airfoils/')
    rel_path = 'airfoils/' + design.application_id + '_' + str(sol_idx) + '.txt'
    abs_path = os.path.join(script_dir, rel_path)
    airfoil.write_coordinates_to_file(abs_path)

    airfoil2 = copy.deepcopy(design.airfoil)

    # Generate the section and write airfoil to file
    airfoil2.generate_section(design.shape_variables_element2, design.n_pts, delta_z_te=delta_z_te)
    rear_element_unscaled = copy.deepcopy(airfoil2)


    # scale the second element
    airfoil2.scale_section(scale_factor=design.scale_element2)
    # rotate the second element
    # TODO remove assumption that rotation occurs around quarter chord
    z_rot = airfoil2.local_camber(x_over_c=[0.25])
    centre_of_rotation = np.array([[0.25], [0.0], [z_rot]])
    x_new, z_new = rotate_element(airfoil2,design.rotation_element2, centre_of_rotation=centre_of_rotation)
    airfoil2.x = x_new
    airfoil2.z = z_new
    # translate the second element
    # TODO add option to specify where the translation occurs - for now asssumed it it relative to the trailing edge
    #  of the front element
    delta_x = airfoil.x[0] + design.translation_element2[0]
    delta_z = (airfoil.z[-1]+airfoil.z[0])/2 + design.translation_element2[1]

    airfoil2.x += delta_x
    airfoil2.z += delta_z


    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists('./airfoils/'):
        os.makedirs('./airfoils/')
    rel_path = 'airfoils/' + design.application_id + '_' + str(sol_idx) + '_element2.txt'
    abs_path = os.path.join(script_dir, rel_path)
    airfoil2.write_coordinates_to_file(abs_path)


    # Initialise objectives and constraints
    objective = np.ones(design.n_obj)
    constraint = np.zeros(design.n_con)
    performance = [0] * design.n_fc

    # Evaluation function
    cfd_eval = None
    if config.settings.solver.lower() in ['su2', 'openfoam']:
        if config.settings.mesher.lower() == 'gmsh':
            raise Exception('this has not been set up yet')
            # cfd_eval = gmsh_util.run_single_element
        elif config.settings.mesher.lower() == 'construct2d':
            raise Exception('construct2d does not work for 2 elements')
    elif config.settings.solver.lower() in ['xfoil', 'xfoil_python']:
        raise Exception('xfoil does not work for 2 elements')
    elif config.settings.solver.lower() == 'mses':
        cfd_eval = mses_util.mses_wrapper_two_element
    else:
        raise Exception('Solver not valid')

    fig, ax = plt.subplots()
    plt.plot(airfoil.x, airfoil.z)
    plt.plot(airfoil2.x, airfoil2.z)
    plt.show()

    # Run thickness checks
    geometry_check(design, airfoil, front_element_unscaled, airfoil2, rear_element_unscaled, objective, constraint)


    # Initialise result array
    result = Result(design.n_fc)

    # Evaluate airfoil performance
    if design.viable:

        if design.application_id == 'Skawinski':
            for fc_idx in range(design.n_fc):

                # CFD evaluation of airfoil performance
                try:
                    current_result = cfd_eval(design, design.flight_condition[fc_idx],
                                              config.settings.solver,
                                              config.settings.n_core,
                                              identifier=sol_idx,
                                              use_python_xfoil=config.settings.use_python_xfoil,
                                              **kwargs)
                except Exception:
                    current_result = None

                # Assign results
                if current_result:
                    # Assign results
                    result.c_l[fc_idx] = current_result.c_l
                    result.c_d[fc_idx] = current_result.c_d
                    result.c_m[fc_idx] = current_result.c_m
                    # Calculate constraints
                    calculate_constraints(design, current_result, constraint, fc_idx)

                else:
                    print('Solver did not converge')
                    for fc_idx_2 in range(fc_idx, design.n_fc):
                        # Dummy values to represent non-converged solution for remaining flight conditions
                        current_result = Result(1)
                        result.c_l[fc_idx_2] = current_result.c_l
                        result.c_d[fc_idx_2] = current_result.c_d
                        result.c_m[fc_idx_2] = current_result.c_m
                        # Calculate constraints here
                        calculate_constraints(design, current_result, constraint, fc_idx_2)
                    break
            calculate_objectives_Skawinski(design, result, objective, design.n_fc-1)
            # set up a case for max lift
            if design.max_lift:
                calculate_max_lift_objective_and_constraint(design, config.settings.solver, config.settings.mesher,
                                                            config.settings.n_core, config.settings.use_python_xfoil,
                                                            sol_idx, objective, constraint, **kwargs)

        else:
            for fc_idx in range(design.n_fc):

                # CFD evaluation of airfoil performance
                try:
                    current_result = cfd_eval(design, design.flight_condition[fc_idx],
                                              config.settings.solver,
                                              config.settings.n_core,
                                              identifier=sol_idx,
                                              use_python_xfoil=config.settings.use_python_xfoil,
                                              **kwargs)
                except Exception:
                    current_result = None

                # Assign results
                if current_result:
                    # Assign results
                    result.c_l[fc_idx] = current_result.c_l
                    result.c_d[fc_idx] = current_result.c_d
                    result.c_m[fc_idx] = current_result.c_m
                    # TODO update to make more robust. For now this assumes that the number of objectives sets the flight conditions for the objectives\
                    # If there are more flight conditions than objectives they are used to set the max lift constraint
                    if fc_idx < design.n_obj:
                        # Calculate objectives here
                        calculate_objectives(design, current_result, objective, fc_idx)
                        # Calculate constraints here
                        calculate_constraints(design, current_result, constraint, fc_idx)
                    else:
                        # last flight condition is the max lift requirement
                        calculate_constraints(design, current_result, constraint, fc_idx)

                else:
                    print('Solver did not converge')
                    for fc_idx_2 in range(fc_idx, design.n_fc):
                        # Dummy values to represent non-converged solution for remaining flight conditions
                        current_result = Result(1)
                        if fc_idx_2 < design.n_obj:
                            # Calculate objectives here
                            calculate_objectives(design, current_result, objective, fc_idx_2)
                            # Calculate constraints here
                            calculate_constraints(design, current_result, constraint, fc_idx_2)
                        else:
                            # last flight condition is the max lift requirement
                            calculate_constraints(design, current_result, constraint, fc_idx_2)
                    break
    else:
        # two additional constraints are cross-over check and max thickness check
        n_base = 2  # cross-over and max thickness
        if design.leading_edge_radius_constraint is not None:
            n_base += 1
        if design.area_constraint is not None:
            n_base += 1
        if design.number_of_allowed_reversals is not None:
            n_base += 1
        n_geom_constraints = n_base + len(design.thickness_constraints[0])
        constraint[n_geom_constraints:] += 1000.0

    # TODO need to add pitching moment constraint here - set up a run at zero angle of attack
    # Normalise objectives if necessary
    normalise_objectives(design, objective)

    print('objective =', objective)
    print('constraint =', constraint)
    print('----------------------------------------')

    if write_quick_history:
        write_obj_cons(objective, constraint)
    if plot:
        plot_geometry(airfoil, **kwargs)

    subprocess.run(['rm -rf ' + abs_path], shell=True)

    return objective, constraint, performance


def geometry_check(design, airfoil, airfoil1, airfoil2, airfoil3, objective, constraint):
    """
    Geometric checks of the airfoil. Calculates cross-over, max thickness and local thickness constraints
    :param design: airfoil geometry is contained in a design instance as design.airfoil
    :param objective: passed so that it can be updated in case of non-viability
    :param constraint: returns the values of the geometric constraint violations
    :return: None. All instances get updated inside the function
    """
    # create the original geometry and calculate thickness
    temp = copy.deepcopy(airfoil1)
    temp2 = copy.deepcopy(airfoil3)

    # unscale for the thickness and cross-over check
    temp.scale_section(1/design.scale_element1)
    temp2.scale_section(1 / design.scale_element2)
    # move element 2 back to the origin
    temp2.x -= np.min(temp2.x)

    temp.calc_thickness_and_camber()
    temp2.calc_thickness_and_camber()

    # Crossover check
    if np.amin(temp.thickness) < 0.0:
        print('Crossover check failed')
        print('np.amin(airfoil_thickness) =', np.amin(temp.thickness))
        constraint[0] += -1000.0 * np.amin(temp.thickness)
        design.viable = False
    if np.amin(temp2.thickness) < 0.0:
        print('Crossover check failed')
        print('np.amin(airfoil_thickness element 2) =', np.amin(temp2.thickness))
        constraint[1] += -1000.0 * np.amin(temp2.thickness)
        design.viable = False

    # Maximum thickness check
    if np.amax(temp.thickness) < design.max_thickness_constraint:
        print('Maximum thickness check failed')
        print('np.amax(airfoil_thickness) =', np.amax(temp.thickness))
        constraint[3] += 10000.0 * (design.max_thickness_constraint - np.amax(temp.thickness))
    if np.amax(temp2.thickness) < design.max_thickness_constraint:
        print('Maximum thickness check failed')
        print('np.amax(airfoil_thickness element 2) =', np.amax(temp2.thickness))
        constraint[4] += 10000.0 * (design.max_thickness_constraint - np.amax(temp2.thickness))

    # local Thickness check
    thick_vals = temp.local_thickness(design.thickness_constraints[0])
    for thick_idx in range(len(thick_vals)):
        if thick_vals[thick_idx] < design.thickness_constraints[1][thick_idx]:
            constraint[5 + thick_idx] += 1000.0 * (design.thickness_constraints[1][thick_idx] - thick_vals[thick_idx])
            design.viable = False
    # local Thickness check second element
    thick_vals = temp2.local_thickness(design.thickness_constraints[0])
    nr_thickness_constraints = len(thick_vals)
    for thick_idx in range(len(thick_vals)):
        if thick_vals[thick_idx] < design.thickness_constraints[1][thick_idx]:
            constraint[5 + nr_thickness_constraints + thick_idx] += \
                1000.0 * (design.thickness_constraints[1][thick_idx] - thick_vals[thick_idx])
            design.viable = False

    # TODO add constraint to check if the elements intersect
    if design.viable:
        poly_front = Polygon(zip(airfoil.x, airfoil.z))
        poly_rear = Polygon(zip(airfoil2.x, airfoil2.z))
        intersects = (poly_front.intersects(poly_rear))

        if intersects:
            # TODO double check if this works
            constraint[2] = 100 * poly_front.intersection(poly_rear).area/poly_rear.area
            design.viable = False
    else:
        constraint[2] = constraint[0] + constraint[1]

    # To prevent zero gradient due to default value of objective functions due to non-viable solutions, add the
    # geometry infeasibility in equal parts to each objective
    if not design.viable:
        objective += constraint[1] / design.n_obj


def calculate_objectives(design, result, objective, fc_idx):
    """
    Calculates the values of the objectives
    :param design: instance of the design object class
    :param result: instance of the result class
    :param objective: values for the objectives
    :param fc_idx: flight condition identifier for the current result
    :return: None - everything gets updated inside the function
    """
    for idx, obj in enumerate(design.objective):
        if obj == 'max_lift_to_drag':
            if result.c_l < 0.0:
                objective[fc_idx] = result.c_l / result.c_d
            else:
                objective[fc_idx] = -result.c_l / result.c_d

            if objective[fc_idx] < -design.unrealistic_value:
                objective[fc_idx] = design.unrealistic_value
            break
        elif obj == 'max_weighted_lift_to_drag':
            if design.obj_weight is None:
                weight = 1.0
            else:
                weight = design.obj_weight[idx]
            # need a special case if optimised for negative lift (happens with some of the propeller airfoils)
            if result.c_l < 0.0:
                objective[idx] += weight * result.c_l / result.c_d
            else:
                objective[idx] += -weight * result.c_l / result.c_d

            if objective[idx] / weight < -design.unrealistic_value:
                objective[idx] = weight * design.unrealistic_value
        elif obj == 'max_lift':
            objective[fc_idx] = -result.c_l
            break
        elif obj == 'weighted_max_lift':
            if design.obj_weight is None:
                weight = 1.0
            else:
                weight = design.obj_weight[idx]
            objective[idx] += -weight * result.c_l


def normalise_objectives(design, objective):
    """
    normalises the objectives in cases of weighted calculations
    :param design: instance of the design class
    :param objective: objective values
    :return:
    """
    for idx, obj in enumerate(design.objective):
        if obj == 'max_weighted_lift_to_drag' or obj == 'weighted_max_lift':
            if design.obj_weight is None:
                weights = np.ones(design.n_fc)
            else:
                weights = design.obj_weight
            objective[idx] /= sum(weights)


def calculate_constraints(design, result, constraint, fc_idx):
    """
    calculates the non-geometric (aerodynamic) constraints
    :param design: instance of the design class
    :param result: instance of the results class
    :param constraint: constraint vector
    :param fc_idx: index of the current flight condition
    :return:
    """
    temp = design.airfoil
    temp.calc_thickness_and_camber()
    n_base = 2 # cross-over and max thickness
    if design.leading_edge_radius_constraint is not None:
        n_base += 1
        temp.calculate_curvature()
        constraint[n_base] = 1000*(design.leading_edge_radius_constraint - temp.leading_edge_radius)
    if design.area_constraint is not None:
        n_base += 1
        temp.calc_area()
        constraint[n_base] = 100 * (design.area_constraint - temp.area)
    if design.number_of_allowed_reversals is not None:
        n_base += 1
        temp.calc_number_of_reversals()
        total_reversals = temp.nr_bottom_reversals + temp.nr_top_reversals
        constraint[n_base] = (total_reversals - 2*design.number_of_allowed_reversals)

    n_geom_constraints = n_base + len(design.thickness_constraints[0])
    # Minimum lift constraints
    if np.abs(result.c_l - design.lift_coefficient[fc_idx]) > 1e-6:
        constraint[n_geom_constraints + fc_idx] = 100.0 * (design.lift_coefficient[fc_idx] - result.c_l)


def calculate_pitching_moment_constraint(design, result, constraint):
    """
    calculate the pitching moment constraint
    TODO needs to be updated and completed once we want to use pitching moment constraints - placeholder for now
    :param design: instance of the design class
    :param result: instance of the results class
    :param constraint: list of all the constraints
    :return: None - everything gets updated inside the function
    """
    constraint[-1] = 100.0 * (design.pitching_moment - result.c_m)


def calculate_objectives_Skawinski(design, result, objective, fc_idx):
    """
    Calculates the values of the objectives
    :param design: instance of the design object class
    :param result: instance of the result class
    :param objective: values for the objectives
    :param fc_idx: flight condition identifier for the current result
    :return: None - everything gets updated inside the function
    """
    for idx, obj in enumerate(design.objective):
        if obj == 'max_weighted_lift_to_drag':
            if design.obj_weight is None:
                weight = 1.0
            else:
                weight = np.array(design.obj_weight)
            objective[idx] = np.sum(-weight * result.c_l / result.c_d)

            if objective[idx]/sum(weight) < -design.unrealistic_value:
                objective[idx] = design.unrealistic_value
        elif obj == 'max_lift':
            objective[idx] = -result.c_l[fc_idx]
            break
        elif obj == 'weighted_max_lift':
            if design.obj_weight is None:
                weight = 1.0
            else:
                weight = design.obj_weight[idx]
            objective[idx] = np.sum(-weight * result.c_l)
        elif obj == 'max_lift_to_drag':
            objective[idx] = np.sum(-result.c_l / result.c_d)/len(result.c_l)
        elif obj == 'max_thickness':
            temp = design.airfoil
            temp.calc_thickness_and_camber()
            objective[idx] = -100*np.amax(temp.thickness)


def calculate_max_lift_objective_and_constraint(design, solver, mesher, n_core, use_python_xfoil, sol_idx,
                                                objective,constraint,**kwargs):
    flight_condition = flight.FlightCondition()
    flight_condition.set(h=0.0, reynolds=design.reynolds[-1], mach=design.mach[-1])
    flight_condition.alpha = design.max_lift_angle[0]
    cfd_eval = None
    if solver.lower() in ['su2', 'openfoam']:
        if mesher.lower() == 'gmsh':
            cfd_eval = gmsh_util.run_single_element
        elif mesher.lower() == 'construct2d':
            cfd_eval = construct2d_util.run_single_element
    elif solver.lower() in ['xfoil', 'xfoil_python']:
        cfd_eval = xfoil_util.xfoil_wrapper_max_lift
    else:
        raise Exception('Solver not set up yet for a max lift case')
    try:
        current_result = cfd_eval(design, flight_condition,
                                  solver,
                                  n_core,
                                  identifier=sol_idx,
                                  use_python_xfoil=use_python_xfoil,
                                  **kwargs)
    except Exception:
        current_result = None
    # Assign results
    result_max_lift = Result(1)
    if current_result:
        # Assign results
        result_max_lift.c_l = current_result.c_l
        result_max_lift.c_d = current_result.c_d
        result_max_lift.c_m = current_result.c_m
        # Calculate constraints
    else:
        print('Solver did not converge')
    # calculate the objective
    for idx, obj in enumerate(design.objective):
        if obj == 'max_lift':
            objective[idx] = -result_max_lift.c_l
    # calculate the constraint
    constraint[-1] = 100.0 * (design.max_lift_constraint - result_max_lift.c_l)

