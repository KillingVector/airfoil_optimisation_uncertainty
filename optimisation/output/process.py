import numpy as np
import pandas as pd
import copy
import os
import pickle
import pickle5
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt

from optimisation.output.util import extract_data
from optimisation.output import plot
from optimisation.output import geometry
from optimisation.output import analysis

def process(test_case, parametrisation_method, nr_design_variables, filename=None, feasible_only=True):

    # Load history
    if filename is not None:
        with open('../../results/' + filename, 'rb') as f:
            try:
                history = pickle.load(f)
            except ValueError:
                history = pickle5.load(f)
            f.close()
    else:
        with open('../../results/optimisation_history_' + test_case + '.pkl', 'rb') as f:
            try:
                history = pickle.load(f)
            except ValueError:
                history = pickle5.load(f)
            f.close()

    # Number of generations
    n_gen = len(history)
    print(n_gen)

    # Pre-allocate pop_data
    n_var = len(history[0][0].var)
    n_obj = len(history[0][0].obj)
    if history[0][0].cons is not None:
        # One additional column for cons_sum
        n_cons = len(history[0][0].cons) + 1
    else:
        # One column for constraint = None and one for cons_sum
        n_cons = 2
    pop_data = np.zeros((0, n_var + n_obj + n_cons + 3))

    # Extract data
    names = []
    for gen_idx in range(n_gen):
        gen_data, names, _ = extract_data(history[gen_idx], gen_idx)
        pop_data = np.concatenate((pop_data, gen_data), axis=0)

    # Forming dataframe
    df = pd.DataFrame(data=pop_data, columns=names)
    df['generation'] = df['generation'].astype(int)

    # Considering only feasible solutions
    if feasible_only:
        df = df[df['cons_sum'] <= 0.0]

    # Initialise problem for (re)-analysis
    setup, opt_prob = analysis.initialise(test_case, parametrisation_method, nr_design_variables)

    # Extract geometry
    extract_geometry = True
    n_geometry = 2
    geom_idx = []
    var_dict = None
    if extract_geometry:

        # Generate indices of selected non-dominated solutions
        # geom_idx = geometry.extract_indices(copy.copy(df), n_gen, n_geometry, n_obj,
        #                                     feasible_only=feasible_only)
        geom_idx = extract_indices(copy.copy(df), n_gen, n_geometry, n_obj, feasible_only=feasible_only)
        # Generate & serialise var_dicts of selected non-dominated solutions
        var_dict = generate_dicts(test_case, opt_prob, df[np.in1d(df['generation'], n_gen - 1)],
                                           indices=geom_idx,
                                           feasible_only=feasible_only)
        # var_dict = geometry.generate_dicts(test_case, opt_prob, df[np.in1d(df['generation'], n_gen - 1)],
        #                                    indices=geom_idx,
        #                                    feasible_only=feasible_only)
    # Plot
    plot_pareto = True
    plot_scatterplot_matrix = False
    if plot_pareto and len(df) > 0:
        if n_gen <2 :
            plot.pareto_front(copy.copy(df), test_case, colour_by_generation=False, indices=geom_idx,
                              feasible_only=feasible_only)
        else:
            plot.pareto_front(copy.copy(df), test_case, colour_by_generation=True, indices=geom_idx,
                              feasible_only=feasible_only)
    if plot_scatterplot_matrix and len(df) > 0:
        plot.scatterplot_matrix(df, test_case, n_gen, colour_by_generation=True, group_generations=True)

    # Re-run objective function
    run_obj_func = True
    if run_obj_func and var_dict is not None:
        obj = np.zeros((len(var_dict), n_obj))
        cons = np.zeros((len(var_dict), n_cons-1))  # No cons-sum here
        performance = [0]*len(var_dict)
        for i in range(len(var_dict)):
            obj[i, :], cons[i, :], performance[i] = analysis.run_obj_func(setup, var_dict[i],
                                                                          idx=i,
                                                                          write_quick_history=False,
                                                                          plot=True,
                                                                          plot_name=test_case + '_idx_' + str(i))

def extract_indices(data, n_gen, n_geometry, n_obj, feasible_only=True):

    # Only consider non-dominated solutions from the final generation
    gen_mask = np.in1d(data['generation'], n_gen - 1)
    non_dominated_mask = (data['rank'] == 0.0)
    if feasible_only:
        mask = gen_mask & non_dominated_mask
    else:
        mask = gen_mask
    data = data[mask]

    # Calculate feasible nadir & zeniths
    feasible_nadir = np.array([np.amax(data['f_0']), np.amax(data['f_1'])])
    feasible_zenith = np.array([np.amin(data['f_0']), np.amin(data['f_1'])])

    if len(data) > 0:

        # Normalise objective function values
        f_0_norm = (data['f_0'].values - feasible_nadir[0]) / (feasible_zenith[0] - feasible_nadir[0])
        f_1_norm = (data['f_1'].values - feasible_nadir[1]) / (feasible_zenith[1] - feasible_nadir[1])

        # Indices for the best solution for each individual objective
        idx_best_f_0 = np.argmax(f_0_norm)
        idx_best_f_1 = np.argmax(f_1_norm)

        # Indices for the highest distance from the feasible nadir point (in the objective space)
        norm_obj_dist = np.sqrt(f_0_norm ** 2.0 + f_1_norm ** 2.0)
        ranked_idx = np.argsort(norm_obj_dist)[::-1]

        # rank based on f_0 or f_1
        # ranked_idx = np.argsort(f_1_norm)[::-1]
        # ranked_idx = np.argsort(f_0_norm)[::-1]


        # Calculate indices to output
        if n_geometry > len(data):
            geom_idx = []
            raise Exception('Number of output geometries must be <= number of non-dominated solutions')
        elif n_geometry == n_obj:
            geom_idx = [idx_best_f_0, idx_best_f_1]
        else:
            geom_idx = [idx_best_f_0, idx_best_f_1]
            ctr = 0
            while len(geom_idx) < n_geometry:
                if ranked_idx[ctr] not in geom_idx:
                    geom_idx.append(ranked_idx[ctr])
                ctr += int(np.floor(len(ranked_idx)/n_geometry))
    else:
        geom_idx = []

    return geom_idx


def generate_dicts(test_case, problem, data, indices, feasible_only=True):

    if feasible_only:
        # Only consider non-dominated solutions from the final generation
        mask = (data['rank'] == 0.0)
        data = data[mask]

    # Extracting data from passed indices
    data = data.iloc[indices]

    # Dropping non-var columns
    if test_case == 'eVTOL':
        data = data.drop(columns=['f_0',
                                'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                'constraint_10', 'constraint_11', 'constraint_12',  'cons_sum', 'constraint_13',
                                'constraint_14', 'constraint_15', 'constraint_16',
                                # 'constraint_14', 'max_lift_angle_0',
                                'rank', 'crowding_distance', 'generation'])
    elif test_case == 'prop_0775r':
        data = data.drop(columns=['f_0','f_1',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  'constraint_14', 'constraint_15',#'constraint_16',
                                  #'constraint_14', #'max_lift_angle_0',
                                  'rank', 'crowding_distance', 'generation'])
    elif test_case == 'prop_0775r_thin':
        data = data.drop(columns=['f_0','f_1',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  'constraint_14', 'constraint_15', #'constraint_16', 'constraint_17',
                                  # 'constraint_14', 'max_lift_angle_0',
                                  'rank', 'crowding_distance', 'generation'])
    elif test_case == 'prop_0995r':
        data = data.drop(columns=['f_0', 'f_1',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  'constraint_14', #'constraint_15',  #'constraint_16',
                                  # 'constraint_14', 'max_lift_angle_0',
                                  'rank', 'crowding_distance', 'generation'])
    else:
        data = data.drop(columns=['f_0','f_1',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  'constraint_14', #'constraint_15', #'constraint_16',
                                  # 'constraint_14', 'max_lift_angle_0',
                                  'rank', 'crowding_distance', 'generation'])



    # var_dict for each solution
    var_dict = [OrderedDict() for _ in range(len(data))]

    for i in range(len(data)):
        # Extract current solution from row
        temp = data.iloc[i].copy(deep=True)

        # Cycle through the variables
        keys = temp.keys()
        extracted = np.zeros(len(keys), dtype=bool)
        for j in range(len(keys)):
            if not extracted[j]:
                # Extract variable at current index
                extracted[j] = True
                ctr = 1

                # Length of key to check for reconstruction of arrays
                key_len = len(keys[0]) - 2

                # Check if reconstruction of numpy array is required
                while j + ctr < len(keys):
                    if keys[j][:key_len] == keys[j + ctr][:key_len]:
                        extracted[j + ctr] = True
                        ctr += 1
                    else:
                        break

                # Extract variables as required
                key = keys[j][:-2]
                if ctr == 1:
                    var_dict[i][key] = np.array([temp[j]])
                else:
                    var_dict[i][key] = temp[j:j+ctr].values

                # De-scale variables
                null = 0
                if problem.variables[key][0].type == 'c':
                    var_dict[i][key] = var_dict[i][key]/problem.variables[key][0].scale
                elif problem.variables[key][0].type == 'i':
                    var_dict[i][key] = round(var_dict[i][key]/problem.variables[key][0].scale)
                elif problem.variables[key][0].type == 'd':
                    idx = np.round(var_dict[i][key]/problem.variables[key][0].scale, 0).astype(int)
                    var_dict[i][key] = np.asarray(problem.variables[key][0].choices)[idx].tolist()

    # Output
    return var_dict


if __name__ == '__main__':
    case_number = 2
    file_name = None

    if case_number == 1:
        # test_case = 'Skawinski'
        # test_case = 'prop_0775r_thin'
        test_case = 'prop_0995r'
        # test_case = 'prop_0505r_thick'

        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 2:
        test_case = 'prop_0775r_thin'
        # test_case = 'prop_0995r'
        # test_case = 'prop_0235r_thin'
        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 3:
        test_case = 'prop_0995r'
        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 4:
        # test_case = 'prop_0505r'
        test_case = 'prop_0235r_thin'
        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 5:
        # test_case = 'Skawinski'
        test_case = 'prop_0235r'
        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 6:
        # test_case = 'Skawinski'
        test_case = 'prop_0995r'
        parametrisation_method = 'Bezier'
        nr_design_variables = 16
    elif case_number == 7:
        # test_case = 'Skawinski'
        test_case = 'prop_0995r'
        parametrisation_method = 'HicksHenne'
        nr_design_variables = 20

    process(test_case, parametrisation_method, nr_design_variables, filename=file_name, feasible_only=True)

