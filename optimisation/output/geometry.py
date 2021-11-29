import numpy as np
import pandas as pd
from collections import OrderedDict
import pickle


def extract_indices(data, n_gen, n_geometry, n_obj, feasible_only=True):

    # Calculate feasible nadir & zeniths
    feasible_nadir = np.array([np.amax(data['f_0']), np.amax(data['f_1'])])
    feasible_zenith = np.array([np.amin(data['f_0']), np.amin(data['f_1'])])

    # Only consider non-dominated solutions from the final generation
    gen_mask = np.in1d(data['generation'], n_gen - 1)
    non_dominated_mask = (data['rank'] == 0.0)
    if feasible_only:
        mask = gen_mask & non_dominated_mask
    else:
        mask = gen_mask
    data = data[mask]

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
                ctr += 1
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
    data = data.drop(columns=['f_0', 'f_1',
                              'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                              'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                              'constraint_10', 'constraint_11', 'constraint_12',  'cons_sum', 'constraint_13',
                              'constraint_14',
                              'rank', 'crowding_distance', 'generation'])

    # var_dict for each solution
    var_dict = [OrderedDict() for _ in range(len(data))]

    for i in range(len(data)):
        # Extract current solution from row
        temp = data.iloc[i].copy(deep=True)

        # Cycle through the variables
        keys = temp.keys()
        extracted = np.zeros(len(keys), dtype=np.bool)
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

    # Serialise var_dicts
    with open('../../results/' + test_case + '_selected_solution_var_dicts' + '.pkl', 'wb') as f:
        pickle.dump(var_dict, f, pickle.HIGHEST_PROTOCOL)

    # Output
    return var_dict

