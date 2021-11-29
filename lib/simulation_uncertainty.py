import numpy as np
import os
import subprocess

from lib import config
from lib.graphics import plot_geometry
from lib.result_uncertainty import Result
from lib.cfd_lib import gmsh_util, construct2d_util, xfoil_util, mses_util
from lib.util import write_obj_cons_uncertainty
from lib import flight

from lib import uncertainty_util as ut

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
    if config.settings.mesher == 'construct2d':
        delta_z_te = 0.0025
    else:
        delta_z_te = 0.0025

    # Generate the section and write airfoil to file
    airfoil.generate_section(design.shape_variables, design.n_pts, delta_z_te=delta_z_te)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists('./airfoils/'):
        os.makedirs('./airfoils/')
    rel_path = 'airfoils/' + design.application_id + '_' + str(sol_idx) + '.txt'
    abs_path = os.path.join(script_dir, rel_path)
    airfoil.write_coordinates_to_file(abs_path)

    # Initialise objectives and constraints
    objective = np.ones(design.n_obj)
    constraint = np.zeros(design.n_con)
    performance = [0] * design.n_fc
    
    # Initialise Uncertainty values
    uncertainty = np.ones((design.n_obj, 6))
    # each row is uncertainty for sp obj
    # col: std, var, skew, kurt, sobol1, sobol2

    # Evaluation function
    cfd_eval = None
    if config.settings.solver.lower() in ['su2', 'openfoam']:
        if config.settings.mesher.lower() == 'gmsh':
            cfd_eval = gmsh_util.run_single_element
        elif config.settings.mesher.lower() == 'construct2d':
            cfd_eval = construct2d_util.run_single_element
    elif config.settings.solver.lower() in ['xfoil', 'xfoil_python']:
        cfd_eval = xfoil_util.xfoil_wrapper
    elif config.settings.solver.lower() == 'mses':
        cfd_eval = mses_util.mses_wrapper
    else:
        raise Exception('Solver not valid')

    # Run thickness checks
    geometry_check(design, objective, constraint)

    # Initialise result array
    result = Result(design.n_fc)

    # Evaluate airfoil performance
    if design.viable:
        if design.application_id == 'Skawinski' or design.application_id == 'eVTOL':
            raise Exception('no bad! no choose skawinski or evtol')
        else:
            for fc_idx in range(design.n_fc):
                # TODO uncertainty setup here - may need calculations
                # uncertainty_util imported as ut
                unc     = config.settings.uncertainty
                if unc.tag.lower() == 'vt':
                    # if design.flight_condition[fc_idx].u == 0.0 or design.flight_condition[fc_idx].ambient.T == 0.0:
                    # TODO - issue: line below is necessary, but if vel and T are set in the design
                    # TODO - then it needs to go the other way. In this case initial design is set with M and Re
                    design.flight_condition[fc_idx].MRe_to_velT()
                    vel = design.flight_condition[fc_idx].u
                    T   = design.flight_condition[fc_idx].ambient.T
                    var = [vel, T]
                elif unc.tag.lower() == 'mre':
                    mach= design.flight_condition[fc_idx].mach
                    re  = design.flight_condition[fc_idx].reynolds
                    var = [mach, re]
                # create nodes
                dist_list           = ut.create_sample(variable = var,
                                                        var_dist    = unc.dist,
                                                        initial_dist= unc.initial,
                                                        order        = unc.order)
                nodes, weights, dist= ut.generate_quadrature(dist_list = dist_list,
                                                            rule        = unc.rule,
                                                            order       = unc.order)
                chk = np.zeros((2,len(nodes[0,:])))
                if unc.tag.lower() == 'vt':
                    for i in range(0, len(nodes[0, :])):
                        v = nodes[0, i]
                        T = nodes[1, i]
                        m = v / np.sqrt(1.4 * 287 * T)
                        Re = 101.325e3 * 1 * v / (287 * 1.7894e-5 * T)
                        # print('M : ', m, '\tRe : ', Re)
                        chk[0, i] = m
                        chk[1, i] = Re
                    mstd = np.std(chk[0,:])/np.mean(chk[0,:])
                    rstd = np.std(chk[1,:])/np.mean(chk[1,:])
                elif unc.tag.lower() == 'mre':
                    for i in range(0, len(nodes[0, :])):
                        m = nodes[0, i]
                        r = nodes[1, i]
                        v = m**2/r * (1.4 * 1 * 101.325e3/1.789e-5)
                        T = m**2/r**2 * (1.4 * 101.325e3**2)/(287*1.789e-5**2)
                        # print('M : ', m, '\tRe : ', Re)
                        chk[0, i] = v
                        chk[1, i] = T
                    vstd = np.std(chk[0,:])/np.mean(chk[0,:])
                    tstd = np.std(chk[1,:])/np.mean(chk[1,:])
                n1std = np.std(nodes[0,:])/np.mean(nodes[0,:])
                n2std = np.std(nodes[1,:])/np.mean(nodes[1,:])

                # get result evaluations
                eval_cl     = []
                eval_cd     = []
                eval_cm     = []
                eval_ld   = []
                # CFD evaluation of airfoil performance
                try:
                    for i,node in enumerate(nodes.T):
                        print("\t\t Node Runs:",i)
                        # allocate nodes      
                        if unc.tag.lower() == 'vt':
                            print('vt: v',node[0],'\tT',node[1])
                            design.flight_condition[fc_idx].u           = node[0]
                            design.flight_condition[fc_idx].ambient.T   = node[1]
                            # convert to equivalen mach and reynolds
                            # design.flight_condition[fc_idx].velT_to_MRe()
                            # design.flight_condition[fc_idx].MRe_to_velT()
                            design.flight_condition[fc_idx].vm_set() # set vel, T, M, Re
                            design.flight_condition[fc_idx].u
                            design.flight_condition[fc_idx].ambient.T
                            block = 0
                        elif unc.tag.lower() == 'mre':
                            print('mr: m',node[0],'\tr',node[1])
                            design.flight_condition[fc_idx].mach    = node[0]
                            design.flight_condition[fc_idx].reynolds= node[1]
                            # convert to equivalent v and T
                            # design.flight_condition[fc_idx].MRe_to_velT()
                            # design.flight_condition[fc_idx].velT_to_MRe()
                            design.flight_condition[fc_idx].mv_set() # set vel, T, M, Re
                            design.flight_condition[fc_idx].mach
                            design.flight_condition[fc_idx].reynolds
                            block = 0
                        current_result = cfd_eval(design, design.flight_condition[fc_idx],
                                                  config.settings.solver,
                                                  config.settings.n_core,
                                                  identifier=sol_idx,
                                        use_python_xfoil=config.settings.use_python_xfoil,
                                                  **kwargs)
                        print("\t\t EVAL SUCCESS")
                        eval_cl.append(current_result.c_l)
                        eval_cd.append(current_result.c_d)
                        eval_cm.append(current_result.c_m)
                        clcd    = current_result.c_l / current_result.c_d
                        eval_ld.append(clcd)
                except Exception as e:
                    print(e)
                    print("\t\t IT FAILED")
                    current_result = None

                if current_result:
                    # Check failures using eval_cl
                    fail_cases  = ut.check_failures(evals=eval_cl, FAIL_CRITERIA=-1)
                    # correct length of nodes and weights
                    # TODO - If len(fail_cases) > 2 * unc.order: assign -1 to mean and skip the rest of the stat process
                    nodes, weights = ut.correct_nodes_weights(nodes=nodes, weights=weights, fail=fail_cases)
                    # correct length of evaluations
                    eval_cl = ut.correct_evals(evals=eval_cl, fail=fail_cases)
                    eval_cd = ut.correct_evals(evals=eval_cd, fail=fail_cases)
                    eval_cm = ut.correct_evals(evals=eval_cm, fail=fail_cases)
                    eval_ld = ut.correct_evals(evals=eval_ld, fail=fail_cases)

                # if too many cases are failed cases, then they cannot be run
                # TODO uncertainty
                if current_result and len(fail_cases) < 2*unc.order+1:
                    model_cl    = ut.create_model(  nodes   = nodes,
                                                    weights = weights,
                                                    evals   = eval_cl,
                                                    joint   = dist,
                                                    order   = unc.order)
                    model_cd    = ut.create_model(  nodes   = nodes,
                                                    weights = weights,
                                                    evals   = eval_cd,
                                                    joint   = dist,
                                                    order   = unc.order)
                    model_cm    = ut.create_model(  nodes   = nodes,
                                                    weights = weights,
                                                    evals   = eval_cm,
                                                    joint   = dist,
                                                    order   = unc.order)
                    model_ld    = ut.create_model(  nodes   = nodes,
                                                    weights = weights,
                                                    evals   = eval_ld,
                                                    joint   = dist,
                                                    order   = unc.order)
                    stats_cl    = ut.get_statistics(model       = model_cl,
                                                    distribution= dist)
                    stats_cd    = ut.get_statistics(model       = model_cd,
                                                    distribution= dist)
                    stats_cm    = ut.get_statistics(model       = model_cm,
                                                    distribution= dist)
                    stats_ld    = ut.get_statistics(model       = model_ld,
                                                    distribution= dist)
                    # stats[0] [mean, std, variance, skew, kurtosis]
                    # stats[1] sobol sensitivity
                elif current_result and len(fail_cases) >= 2*unc.order+1:
                    current_result = None

                # Assign results
                if current_result:
                    # Assign results
                    # set current result into uncertainty format
                    current_result = set_current_result(stats_cl, stats_cd, stats_cm, stats_ld)

                    result  = set_result(fc_idx, current_result, result)

                    # TODO update to make more robust. For now this assumes that the number of objectives sets the flight conditions for the objectives\
                    # If there are more flight conditions than objectives they are used to set the max lift constraint
                    if fc_idx < design.n_obj:
                        # Calculate objectives here
                        calculate_objectives(design, current_result, objective, uncertainty, fc_idx)
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
                            calculate_objectives(design, current_result, objective, uncertainty, fc_idx_2)
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
    print('----------------------------------------')
    normalise_objectives(design, objective)

    print('objective =', objective)
    print('constraint =', constraint)
    print('std deviation = ', uncertainty)
    print('----------------------------------------')

    if write_quick_history:
        write_obj_cons_uncertainty(objective, constraint, uncertainty)
    if plot:
        plot_geometry(airfoil, **kwargs)

    subprocess.run(['rm -rf ' + abs_path], shell=True)
    return objective, constraint, performance


def geometry_check(design, objective, constraint):
    """
    Geometric checks of the airfoil. Calculates cross-over, max thickness and local thickness constraints
    :param design: airfoil geometry is contained in a design instance as design.airfoil
    :param objective: passed so that it can be updated in case of non-viability
    :param constraint: returns the values of the geometric constraint violations
    :return: None. All instances get updated inside the function
    """

    # create the geometry and calculate thickness
    temp = design.airfoil
    temp.calc_thickness_and_camber()
    # Crossover check
    if np.amin(temp.thickness) < 0.0:
        print('Crossover check failed')
        print('np.amin(airfoil_thickness) =', np.amin(temp.thickness))
        # TODO might need to remove the scaling factors when surrogates are used - to be checked
        #  (not needed if normalised in a sklearn pipeline)
        constraint[0] += -1000.0 * np.amin(temp.thickness)
        design.viable = False

    # Maximum thickness check
    if np.amax(temp.thickness) < design.max_thickness_constraint:
        print('Maximum thickness check failed')
        print('np.amax(airfoil_thickness) =', np.amax(temp.thickness))
        # TODO might need to remove the scaling factors when surrogates are used - to be checked
        #  (not needed if normalised in a sklearn pipeline)
        constraint[1] += 10000.0 * (design.max_thickness_constraint - np.amax(temp.thickness))

    # Maximum thickness check - allow slightly thinner airfoils
    if np.amax(temp.thickness) < 0.8 * design.max_thickness_constraint:
        print('Maximum thickness less than 80% of required thickness - set as a non-viable geometry')
        # TODO might need to remove the scaling factors when surrogates are used - to be checked
        #  (not needed if normalised in a sklearn pipeline)
        constraint[1] += 10000.0 * (design.max_thickness_constraint - np.amax(temp.thickness))
        design.viable = False

    # # TODO for now this is only set up for Skawinski
    if np.amax(temp.thickness) > design.max_thickness_margin*design.max_thickness_constraint:
        print('Maximum thickness check failed')
        print('np.amax(airfoil_thickness) =', np.amax(temp.thickness))
        constraint[1] += 10000.0 * (np.amax(temp.thickness) -
                                    design.max_thickness_margin*design.max_thickness_constraint)

    # local Thickness check
    thick_vals = temp.local_thickness(design.thickness_constraints[0])
    for thick_idx in range(len(thick_vals)):
        if thick_vals[thick_idx] < design.thickness_constraints[1][thick_idx]:
            # TODO might need to remove the scaling factors when surrogates are used - to be checked
            #  (not needed if normalised in a sklearn pipeline)
            constraint[2 + thick_idx] += 1000.0 * (design.thickness_constraints[1][thick_idx] - thick_vals[thick_idx])
            design.viable = False

    # To prevent zero gradient due to default value of objective functions due to non-viable solutions, add the
    # geometry infeasibility in equal parts to each objective
    if not design.viable:
        objective += constraint[1] / design.n_obj


def calculate_objectives(design, result, objective, uncertainty, fc_idx):
    """
    Calculates the values of the objectives
    !! UNCERTAINTY: uses statistical expected value for L/D rather than calc
    :param design: instance of the design object class
    :param result: instance of the result class
    :param objective: values for the objectives
    :param fc_idx: flight condition identifier for the current result
    :return: None - everything gets updated inside the function
    """
    unc = config.settings.uncertainty

    for idx, obj in enumerate(design.objective):
        # add stat modes to uncertainty structure
        uncertainty[fc_idx, :] = np.concatenate((result.ld_stats, result.ld_sens), axis=0)

        if obj == 'max_lift_to_drag':
            # objective
            if result.c_l < 0.0:
                objective[idx] += (result.l_d + design.uncertainty_weight * result.ld_stats[0])#result.c_l / result.c_d
            else:
                objective[idx] += -(result.l_d - design.uncertainty_weight * result.ld_stats[0])#result.c_l / result.c_d

            # if objective[fc_idx] < -design.unrealistic_value:
            if objective[fc_idx] + np.abs(unc.sigma*result.ld_stats[0]) > -design.unrealist_value:
                objective[fc_idx] = design.unrealistic_value
            break
        elif obj == 'max_weighted_lift_to_drag':
            # objective
            if design.obj_weight is None:
                weight = 1.0
            else:
                weight = design.obj_weight[idx]
            # need a special case if optimised for negative lift (happens with some of the propeller airfoils)
            if result.c_l < 0.0:
                objective[idx] += weight * (result.l_d + design.uncertainty_weight * result.ld_stats[0])#result.c_l / result.c_d
            else:
                objective[idx] += -weight * (result.l_d - design.uncertainty_weight * result.ld_stats[0])#result.c_l / result.c_d

            if (objective[idx]+np.abs(unc.sigma * result.ld_stats[0])) / weight < -design.unrealistic_value:
                objective[idx] = weight * design.unrealistic_value
        elif obj == 'max_lift':
            # objective
            # objective[fc_idx] = -result.c_l
            if result.c_l < 0.0:
                objective[fc_idx] = -result.c_l + design.uncertainty_weight * result.cl_stats[0]
            else:
                objective[fc_idx] = -result.c_l - design.uncertainty_weight * result.cl_stats[0]
            break
        elif obj == 'weighted_max_lift':
            # objective
            if design.obj_weight is None:
                weight = 1.0
            else:
                weight = design.obj_weight[idx]
            if result.c_l < 0.0:
                objective[fc_idx] = -weight * (result.c_l + design.uncertainty_weight * result.cl_stats[0])
            else:
                objective[fc_idx] = -weight * (result.c_l - design.uncertainty_weight * result.cl_stats[0])
            # objective[idx] += -weight * result.c_l




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
    unc     = config.settings.uncertainty

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
        if temp.nr_bottom_reversals < design.number_of_allowed_reversals:
            if temp.nr_top_reversals < design.number_of_allowed_reversals:
                constraint[n_base] = (total_reversals - 2*design.number_of_allowed_reversals)
            else:
                constraint[n_base] = (temp.nr_top_reversals - design.number_of_allowed_reversals)
        else:
            constraint[n_base] = (temp.nr_bottom_reversals - design.number_of_allowed_reversals)

    n_geom_constraints = n_base + len(design.thickness_constraints[0])
    # Minimum lift constraints
#    if np.abs(result.c_l - design.lift_coefficient[fc_idx]) > 1e-6:
    # UNCERTAINTY APPLIED HERE: TODO - include method of changing sigma value
    if np.abs((result.c_l - unc.sigma * result.cl_stats[0]) - design.lift_coefficient[fc_idx]) > 1e-6:
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
    unc = config.settings.uncertainty
    # max or min?
    cm = max(np.abs(result.c_m - unc.sigma * result.cm_stats[0]), np.abs(result.c_m + result.cm_stats[0]))
#    constraint[-1] = 100.0 * (design.pitching_moment - cm)
    constraint[-1] = 100.0 * (design.pitching_moment - result.c_m)


## uncertainty utility
def set_current_result(stats_cl, stats_cd, stats_cm, stats_ld):
    current_result = Result(1)
    ll = len(stats_cl[0])
    current_result.c_l = stats_cl[0][0]
    current_result.c_d = stats_cd[0][0]
    current_result.c_m = stats_cm[0][0]
    current_result.l_d = stats_ld[0][0]
    current_result.cl_stats = stats_cl[0][1:ll]
    current_result.cd_stats = stats_cd[0][1:ll]
    current_result.cm_stats = stats_cm[0][1:ll]
    current_result.ld_stats = stats_ld[0][1:ll]
    current_result.cl_sens = stats_cl[1]
    current_result.cd_sens = stats_cd[1]
    current_result.cm_sens = stats_cm[1]
    current_result.ld_sens = stats_ld[1]
    return current_result

def set_result(fc_idx, current_result, result):
    # Assign results
    result.c_l[fc_idx] = current_result.c_l
    result.c_d[fc_idx] = current_result.c_d
    result.c_m[fc_idx] = current_result.c_m
    result.l_d[fc_idx] = current_result.l_d
    result.cl_stats[fc_idx, :] = current_result.cl_stats
    result.cd_stats[fc_idx, :] = current_result.cd_stats
    result.cm_stats[fc_idx, :] = current_result.cm_stats
    result.ld_stats[fc_idx, :] = current_result.ld_stats
    result.cl_sens[fc_idx, :] = current_result.cl_sens
    result.cd_sens[fc_idx, :] = current_result.cd_sens
    result.cm_sens[fc_idx, :] = current_result.cm_sens
    result.ld_sens[fc_idx, :] = current_result.ld_sens
    return result