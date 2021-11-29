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


from matplotlib import font_manager
font_dirs = ['/Users/3s/Library/Fonts' ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Gulliver-Regular'


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('legend', fontsize=11)


myblack = [0, 0, 0]
myblue = '#0F95D7'
myred = [220 / 255, 50 / 255, 32/ 255]
myyellow = [255 / 255, 194 / 255, 10 / 255]
# mygreen = [64/255, 176 / 255, 166/255]
# mygreen = [65/255, 141 / 255, 43/255]
mygreen = [139/255, 195 / 255, 74/255]
mybrown = [153/255,79/255,0/255]
mydarkblue = [60/255, 105/255, 225/255]
mypurple = [0.4940, 0.1840, 0.5560]
myorange = [230/255, 97/255, 0/255]
mygray = [89 / 255, 89 / 255, 89 / 255]


colors = [myblack, myblue, myred, myyellow,
            mygreen, mybrown, mydarkblue, mypurple, myorange, mygray]

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
    setup, opt_prob = analysis.initialise_single_objective(test_case, parametrisation_method, nr_design_variables)

    # Extract geometry
    extract_geometry = True
    n_geometry = 0
    geom_idx = []
    var_dict = None
    if extract_geometry:

        # Generate indices of selected non-dominated solutions
        geom_idx = extract_indices(copy.copy(df), n_gen, n_geometry, n_obj,
                                            feasible_only=feasible_only)

        # Generate & serialise var_dicts of selected non-dominated solutions
        var_dict = generate_dicts(test_case, opt_prob, df[np.in1d(df['generation'], n_gen - 1)],
                                           indices=geom_idx,
                                           feasible_only=feasible_only)

    # Plot
    f_obj_0_scale = 1.0
    f_obj_1_scale = -1.0

    data = copy.copy(df)

    x_data = data['generation']
    y_data = data['f_0']

    mpl.rc('savefig', dpi=200, format='pdata', bbox='tight')
    ax_0_label = 'Generations'
    ax_1_label = 'Weighted lift-to-drag'

    fig, ax = plt.subplots()
    plt.scatter(x_data, y_data *f_obj_1_scale, color=colors[0], alpha=1,
                    label='Non-dominated solutions')
    ax.set_xlabel(ax_0_label)
    ax.set_ylabel(ax_1_label)
    ax.grid(True)
    ax.set_ylim(bottom=0.)
    # plt.ylim(0, 100)
    # plt.ylim(1.4, 2.4)
    if np.sign(f_obj_0_scale) == -1:
        ax.invert_xaxis()
    if np.sign(f_obj_1_scale) == -1:
        ax.invert_yaxis()
    plt.show()
    plt.close()

    x_data, y_data = extract_best(data,feasible_only=feasible_only)
    y_data *= f_obj_1_scale



    fig, ax = plt.subplots()
    plt.scatter(x_data, y_data, color=colors[0], alpha=1,
                    label='Non-dominated solutions')
    ax.set_xlabel(ax_0_label)
    ax.set_ylabel(ax_1_label)
    ax.grid(True)
    # plt.xlim(10, 60)
    # plt.ylim(1.4, 2.4)
    if np.sign(f_obj_0_scale) == -1:
        ax.invert_xaxis()
    if np.sign(f_obj_1_scale) == -1:
        ax.invert_yaxis()
    plt.show()
    plt.close()

    # Re-run objective function
    run_obj_func = True
    if run_obj_func and var_dict is not None:
        obj = np.zeros((len(var_dict), n_obj))
        cons = np.zeros((len(var_dict), n_cons-1))  # No cons-sum here
        performance = [0]*len(var_dict)
        for i in range(len(var_dict)):
            obj[i, :], cons[i, :], performance[i] = analysis.run_obj_func_single_objective(setup, var_dict[i],
                                                                          write_quick_history=False,
                                                                          plot=True,
                                                                          plot_name=test_case + '_idx_' + str(i))


def extract_indices(data, n_gen, n_geometry, n_obj, feasible_only=True):

    # Calculate feasible nadir & zeniths
    feasible_nadir = np.array([np.amax(data['f_0'])])
    feasible_zenith = np.array([np.amin(data['f_0'])])

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

        # Indices for the best solution for each individual objective
        idx_best_f_0 = np.argmax(f_0_norm)

        # Indices for the highest distance from the feasible nadir point (in the objective space)
        norm_obj_dist = np.sqrt(f_0_norm ** 2.0)
        ranked_idx = np.argsort(norm_obj_dist)[::-1]

        # Calculate indices to output
        if n_geometry > len(data):
            geom_idx = []
            raise Exception('Number of output geometries must be <= number of non-dominated solutions')
        elif n_geometry == n_obj:
            geom_idx = [idx_best_f_0]
        else:
            geom_idx = [idx_best_f_0]
            ctr = 0
            while len(geom_idx) < n_geometry:
                if ranked_idx[ctr] not in geom_idx:
                    geom_idx.append(ranked_idx[ctr])
                ctr += 1
    else:
        geom_idx = []

    return geom_idx


def extract_best(data, feasible_only=True):

    x_data = data['generation']
    y_data = data['f_0']
    n_gen = np.amax(x_data)

    # Calculate feasible nadir & zeniths
    feasible_nadir = np.array([np.amax(y_data)])
    feasible_zenith = np.array([np.amin(y_data)])

    # Only consider non-dominated solutions from the final generation
    output_x = []
    output_y = []
    for cntr in range(int(n_gen)):
        data2 = copy.deepcopy(data)
        gen_mask = np.in1d(x_data, cntr)
        non_dominated_mask = (data2['rank'] == 0.0)
        if feasible_only:
            mask = gen_mask & non_dominated_mask
        else:
            mask = gen_mask
        data2 = data2[mask]
        if len(data2) > 0:
            best_idx = np.argmin(data2['f_0'])
            best_value = data2['f_0'].values[best_idx]
            output_x.append(cntr)
            output_y.append(best_value)


    return np.array(output_x), np.array(output_y)


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
        data = data.drop(columns=['f_0',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  'constraint_14', 'constraint_15',#'constraint_16',
                                  #'constraint_14', #'max_lift_angle_0',
                                  'rank', 'crowding_distance', 'generation'])
    elif test_case == 'prop_0775r_thin':
        data = data.drop(columns=['f_0',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  'constraint_14', #'constraint_15', #'constraint_16', 'constraint_17',
                                  # 'constraint_14', 'max_lift_angle_0',
                                  'rank', 'crowding_distance', 'generation'])
    elif test_case == 'prop_0995r':
        data = data.drop(columns=['f_0',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  'constraint_14', 'constraint_15',  'constraint_16',
                                  # 'constraint_14', 'max_lift_angle_0',
                                  'rank', 'crowding_distance', 'generation'])
    elif test_case in ['utest','utest_unc','utest_unc2']:
        data = data.drop(columns=['f_0',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', 'cons_sum',
                                  'rank', 'crowding_distance', 'generation'])
    else:
        data = data.drop(columns=['f_0',
                                  'constraint_0', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
                                  'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8', 'constraint_9',
                                  'constraint_10', #'constraint_11', 'constraint_12', 'cons_sum', 'constraint_13',
                                  #'constraint_14', 'constraint_15', #'constraint_16',
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
                # if key == 'cons_s':
                #     break
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

def write_airfoil_dat(filename = None, test_cast = 'test', num_vars = 10, single = True):
    if not filename == None:
        fullfile = '../../results/'+filename
        with open(fullfile,'rb') as f:
            try:
                hist = pickle.load(f)
            except:
                hist = pickle5.load(f)
        last = hist[-1]
    if single: # unnecessary, just returns 0
        # opt = last[0].obj[0]
        # opt_i = 0
        # for i,ll in enumerate(last):
        #     if ll.obj[0] < opt:
        #         opt = ll.obj[0]
        #         opt_i = i
        from lib.airfoil_parametrisation import CSTmod
        airfoil = CSTmod()
        airfoil.generate_section(last[0].var[0:num_vars])
        x = airfoil.x
        z = airfoil.z
        with open('../../airfoils/'+test_case+'.dat','w') as f:
            f.write(test_case.upper() + '\n')
            for i in range(0,len(x)):
                f.write(f'{x[i]:.5f}\t{z[i]:.5f}')
                if i < len(x)-1:
                    f.write('\n')





if __name__ == '__main__':

    # test_case = 'airfoil_testing_xfoil'


    # 1 and 2 are tron3 / 3 & 4 are tron7
    case_number = 0.9

    # file_name = 'xfoil_results/optimisation_history_Skawinski_Parsec_11_shamode_tor_xfoil_attempt_2.pkl'
    # file_name = 'optimisation_history_prop_0995r_CSTmod_19_attempt_0.pkl'
    # file_name = 'optimisation_history_prop_0775r_thin_CST_12_attempt_final_SST.pkl'
    file_name =  'optimisation_history_utest_CSTmod_13_attempt_0.pkl'

    # file_name = None
    # test_case = 'prop_4b_2pos_235rad_21pct'

    if case_number == 0:
        file_name = 'optimisation_history_utest_CSTmod_13_attempt_0.pkl'
        test_case = 'utest'
        parametrisation_method = 'CSTmod'
        nr_design_variables = 13
    elif case_number == 0.5:
        file_name = 'optimisation_history_utest_unc_CSTmod_13_attempt_1.pkl'
        test_case = 'utest_unc'
        parametrisation_method = 'CSTmod'
        nr_design_variables = 13
    elif case_number == 0.9:
        file_name = 'optimisation_history_utest_unc2_CSTmod_13_attempt_2.pkl'
        test_case = 'utest_unc2'
        parametrisation_method = 'CSTmod'
        nr_design_variables = 13

    elif case_number == 1:
        # test_case = 'Skawinski'
        test_case = 'prop_0775r_thin'
        test_case = 'prop_0775r'
        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 2:
        test_case = 'prop_0775r'
        # test_case = 'prop_0995r'
        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 3:
        test_case = 'prop_0995r'
        parametrisation_method = 'CST'
        nr_design_variables = 12
    elif case_number == 4:
        test_case = 'prop_0505r'
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

    process(test_case, parametrisation_method, nr_design_variables, filename=file_name, feasible_only=False)

    # only set up to retrieve single var opt
    write_airfoil_dat(filename=file_name, test_cast=test_case, num_vars=nr_design_variables)
