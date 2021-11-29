import numpy as np
import pandas as pd
import copy, glob
import pickle
import pickle5
import string
from optimisation.output.util import extract_data
from optimisation.output import plot
from optimisation.output import geometry
from optimisation.output import analysis
import matplotlib as mpl
import matplotlib.pyplot as plt

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

def process_multiple(test_case, parametrisation_method, nr_design_variables, algorithm, survival_method, feasible_only=True):

    # Load history
    filestr = '../../results/optimisation_history_' + test_case + '_' + parametrisation_method + '_' + str(nr_design_variables) + '_' \
              + algorithm + '_' + survival_method +'*.pkl'

    filelist = glob.glob(filestr)

    mpl.rc('savefig', dpi=200, format='pdata', bbox='tight')
    labels = list(string.ascii_uppercase)
    ax_0_label = 'Weighted lift-to-drag'
    ax_1_label = 'Max. lift coefficient'
    f_obj_0_scale = -1.0
    f_obj_1_scale = -1.0

    fig, ax = plt.subplots()

    for cntr in range(len(filelist)):
        filename = filelist[cntr]
        with open(filename, 'rb') as f:
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
        df['generation'] = df['generation'].astype(np.int)

        # Considering only feasible solutions
        if feasible_only:
            df = df[df['cons_sum'] <= 0.0]

        # Initialise problem for (re)-analysis
        setup, opt_prob = analysis.initialise(test_case, parametrisation_method, nr_design_variables)

        # Plot
        if len(df) > 0:
            x_data, y_data = extract_pareto(copy.copy(df), feasible_only=feasible_only)

        plt.scatter(x_data, y_data, color=colors[cntr], alpha=1,
                    label='Non-dominated solutions')
    ax.set_xlabel(ax_0_label)
    ax.set_ylabel(ax_1_label)
    ax.grid(True)
    plt.xlim(10, 60)
    plt.ylim(1.4, 2.4)
    if np.sign(f_obj_0_scale) == -1:
        ax.invert_xaxis()
    if np.sign(f_obj_1_scale) == -1:
        ax.invert_yaxis()
    plt.show()
    plt.close()


def extract_pareto(data, feasible_only=True):

    # Scaling objectives
    pd.options.mode.chained_assignment = None   # default='warn'
    f_obj_0_scale = -1.0
    f_obj_1_scale = -1.0
    data['f_0'] *= f_obj_0_scale
    data['f_1'] *= f_obj_1_scale
    # Generations
    generations = data['generation'].unique()

    n_gen = generations[-1]

    # Redoing things so that flipping the final generation dataframe generates the correct results
    gen_mask = np.in1d(data['generation'], n_gen)
    gen_data = data[gen_mask]
    gen_data = gen_data.iloc[::-1]
    non_dominated_mask = (gen_data['rank'] == 0.0)
    if feasible_only:
        non_dominated_data = gen_data[non_dominated_mask]
    else:
        non_dominated_data = gen_data

    x_data = non_dominated_data['f_0'].values
    y_data = non_dominated_data['f_1'].values
    return x_data, y_data



if __name__ == '__main__':

    # test_case = 'airfoil_testing_xfoil'
    test_case = 'Skawinski'
    survival_method = 'cdp'
    algorithm = 'shamode'

    file_name = 'optimisation_history_Skawinski_Bspline_10_shamode_tor_attempt_2.pkl'
    file_name = None
    # test_case = 'prop_4b_2pos_235rad_21pct'
    # parametrisation_method = 'Bspline'
    parametrisation_method = 'Bezier'
    # parametrisation_method = 'HicksHenne'
    parametrisation_method = 'CSTmod'
    nr_design_variables = 13

    process_multiple(test_case, parametrisation_method, nr_design_variables, algorithm, survival_method,
                     feasible_only=True)

