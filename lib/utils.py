import os
import pickle
import pickle5

import numpy as np
from scipy.interpolate import interp1d

import lib.solver_settings as solver_settings
from lib import config
from lib.airfoil_parametrisation import CST, CSTmod, BezierAirfoil, BsplineAirfoil, HicksHenneAirfoil, ParsecAirfoil
from lib.cfd_lib import xfoil_util
from lib.result import Result2 as Result
from lib.settings import Settings
from optimisation.output.util import extract_var


def setup_solver(solver_name, config):
    """
    set up the settings for the solver. This gives a list of defaults that can be updated easily
    :param solver_name: string with the name of the solver
    :param config: config instance for the setup of the optimisation
    :return: none
    """
    if solver_name.lower() == 'xfoil':
        config.settings = Settings(n_core=1, mesher=None, solver='xfoil', use_python_xfoil=False)
    elif solver_name.lower() == 'xfoil_python':
        config.settings = Settings(n_core=1, mesher=None, solver='xfoil', use_python_xfoil=True)
    elif solver_name.lower() == 'openfoam':
        config.settings = Settings(n_core=1, mesher='construct2d', solver='openfoam')
    elif solver_name.lower() == 'su2':
        config.settings = Settings(n_core=32, mesher='construct2d', solver='su2')
    elif solver_name.lower() == 'mses':
        config.settings = Settings(n_core=1, mesher=None, solver='mses')
    elif solver_name.lower() == 'openfoam_gmsh':
        config.settings = Settings(n_core=1, mesher='gmsh', solver='openfoam')
    elif solver_name.lower == 'su2_gmsh':
        config.settings = Settings(n_core=32, mesher='gmsh', solver='su2')

def setup_solver_sweep(solver_name, config):
    """
    set up the settings for the solver. This gives a list of defaults that can be updated easily
    :param solver_name: string with the name of the solver
    :param config: config instance for the setup of the optimisation
    :return: none
    """

    if solver_name.lower() == 'mses':
        config.settings = Settings(n_core=1, mesher=None, solver='mses')
    else:
        NameError('a sweep case is only set up for MSES for now')


def set_coefficient_bounds(parametrisation, n_var):
    # construct the file to read seed airfoils
    if parametrisation == 'CSTmod':
        airfoil_var = (n_var - 1) / 2 - 1
        file_part3 = str(int(airfoil_var))
    elif parametrisation == 'CST':
        airfoil_var = (n_var - 2) / 2
        file_part3 = str(int(airfoil_var))
    else:
        file_part3 = str(n_var)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_path = 'lib/SeedAirfoils/' + parametrisation + file_part3 + '_Seed_Airfoils.csv'
    abs_path = os.path.join(script_dir, rel_path)

    # Read file & extract upper and lower bounds
    coefficients = read_seed_file(abs_path)
    lower_bound = np.amin(coefficients, axis=0)
    upper_bound = np.amax(coefficients, axis=0)

    # Seed coefficient values
    initial_guess = (lower_bound + upper_bound)/2.0
    if parametrisation == 'CST':
        initial_guess = np.array([0.1875261813402176, 0.257924348115921, 0.19586224853992462,
                                  0.21902954578399655, 0.25097155570983887, 0.17126670479774472,
                                  -0.0889078825712204, -0.02791370078921318, -0.046297188848257065,
                                  0.005151660647243261, -0.005898060277104378, 0.09548211097717285])
    elif parametrisation =='Bezier':
        if n_var == 16:
            initial_guess = np.array([0.032062743, 0.106072254, 0.092214624, 0.330877476,
                                     -0.042830002, 0.202035521, 0.038400853, 0.050671418,
                                     -0.046319877, 0.039172494, -0.11513221, 0.11146923,
                                     0.055633316, 0.074114099, 0.07405196, 0.027712414])
        elif n_var == 20:
            initial_guess = np.array([0.019423663, 0.105684052, 0.065018117, 0.161270516,
                                      0.378903832, -0.261437652, 0.458226503, -0.112771885,
                                      0.128273659, 0.018647418, -0.035943881, -0.003816404,
                                      0.008918604, -0.138352186, 0.262818962, -0.137955056,
                                      0.221451838, -0.004259004, 0.078127116, 0.014679128])
        elif n_var == 18:
            initial_guess = np.array([0.016047337, 0.159632977, -0.097963688, 0.604082641,
                                      -0.306731168, 0.417224757,-0.043864969, 0.106671501,
                                      0.030094287, -0.042621146, 0.028399905, -0.08990308,
                                      0.061678623, 0.040443369, 0.099601248, 0.053614815,
                                      0.07122686, 0.020900445])
    else:
        initial_guess = (lower_bound + upper_bound) / 2.0

    return lower_bound, upper_bound, initial_guess


def normalise_seed_airfoils(design):
    # construct the file to read seed airfoils
    filename = generate_seed_filename(design.parametrisation_method, design.n_var)
    coefficients = read_seed_file(filename)

    # normalise the seed airfoils and store them in a new file
    lower_bounds = np.amin(coefficients, axis=0)
    upper_bounds = np.amax(coefficients, axis=0)
    normalised_coefficients = (coefficients - lower_bounds) / (upper_bounds - lower_bounds)
    with open('./lib/normalised_seeds.pkl', 'wb') as f:
        pickle.dump(normalised_coefficients, f, pickle.HIGHEST_PROTOCOL)


def generate_seed_filename(parametrisation_method, n_var):
    file_part1 = 'lib/SeedAirfoils/'
    file_part2 = parametrisation_method
    if parametrisation_method == 'CSTmod':
        if n_var % 2 == 0:
            n_var = (n_var - 2) / 2 - 1
            file_part3 = str(int(n_var))
            new_str = 'A'
            file_part3 = file_part3 + new_str
        else:
            n_var = (n_var - 1) / 2 - 1
            file_part3 = str(int(n_var))
    elif parametrisation_method == 'CST':
            n_var = (n_var - 2) / 2
            file_part3 = str(int(n_var))
    else:
        file_part3 = str(n_var)
    file_part4 = '_Seed_Airfoils.csv'
    filename = file_part1 + file_part2 + file_part3 + file_part4
    return filename


def read_seed_file(filename):
    coefficients, _ = read_seed_file_with_names(filename)
    return coefficients


def read_seed_file_with_names(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    number_of_airfoils = len(lines)
    number_of_coeffs = len(lines[0].split(',')) - 1

    names = []
    coefficients = np.empty([0, 0])
    with open(filename, 'r') as f:
        for line in f:
            fields = line.split(',')
            names.append(fields[0])
            coefficients = np.append(coefficients, np.array(fields[1:], dtype=np.float32))
    coefficients = np.reshape(coefficients, (number_of_airfoils, number_of_coeffs))
    return coefficients, names


def calculate_gradients(x_coords, y_values):
    top_surface = []
    bottom_surface = []

    le_idx = np.argmin(x_coords)
    top_x = x_coords[0:le_idx + 1]
    bot_x = x_coords[le_idx:]
    top_curv = y_values[0:le_idx + 1]
    bot_curv = y_values[le_idx:]

    grad_top_curv = np.gradient(top_curv, top_x)
    grad_bot_curv = np.gradient(bot_curv, bot_x)

    top_surface[0] = top_x
    top_surface[1] = grad_top_curv

    bottom_surface[0] = bot_x
    bottom_surface[1] = grad_bot_curv

    return top_surface, bottom_surface


def generate_variables_from_xfoilrun(filename):
    with open(filename, 'rb') as f:
        try:
            history = pickle.load(f)
        except ValueError:
            history = pickle5.load(f)
        f.close()

    data = history[-1]
    values, _, _ = extract_var(data)

    return values


def airfoil_analysis_xfoil(design, shape_variables, sol_idx, **kwargs):

    airfoil = design.airfoil
    if config.settings.mesher == 'construct2d':
        delta_z_te = 0.0025
    else:
        delta_z_te = 0.0025

    # Writing airfoil to file
    airfoil.generate_section(shape_variables, design.n_pts, delta_z_te=delta_z_te)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_path = 'airfoils/' + design.application_id + '_' + str(sol_idx) + '.txt'
    abs_path = os.path.join(script_dir, rel_path)
    airfoil.write_coordinates_to_file(abs_path)

    # Evaluation function
    cfd_eval = xfoil_util.xfoil_wrapper
    path = solver_settings.XFOIL_PATH
    design.viable = True

    # Initialise result array
    result = Result(design.n_fc)

    # Evaluate airfoil performance
    for fc_idx in range(design.n_fc):

        # CFD evaluation of airfoil performance
        try:
            current_result = cfd_eval(path, design, design.flight_condition[fc_idx],
                                      'xfoil',
                                      config.settings.n_core,
                                      identifier=sol_idx,
                                      use_python_xfoil=True,
                                      **kwargs)
        except Exception:
            current_result = None

            # Assign results
        if current_result:
            # Assign results
            result.c_l[fc_idx] = current_result.c_l
            result.c_d[fc_idx] = current_result.c_d
            result.c_m[fc_idx] = current_result.c_m
            result.alpha[fc_idx] = current_result.alpha

        else:
            print('Solver did not converge')

    output = result.alpha
    return output

def shape_match(control_pts, *args):
    nr_pts = args[0]['npts']
    ta = args[0]['target_airfoil']
    section = args[0]['section']
    with_TE = args[0]['with_TE']
    if with_TE:
        delta_z_te = control_pts[-1]
        control_pts = np.delete(control_pts,-1)
    else:
        delta_z_te = args[0]['delta_z_te']
    if isinstance(section, CST):
        section.generate_section(control_pts, nr_pts, delta_z_te=delta_z_te)
        y_coords = airfoil_resplining2(ta.x, ta.z, section.x)
        caz = section.z
        try:
            return 1000 * np.sum(abs(caz - y_coords))
        except:
            return 1e10
    elif isinstance(section, CSTmod):
        if len(control_pts) % 2 == 1:
            delta_z_te = 0
        else:
            delta_z_te = control_pts[-1]
            control_pts = np.delete(control_pts, -1)
        section.generate_section(control_pts, nr_pts, delta_z_te)
        y_coords = airfoil_resplining2(ta.x, ta.z, section.x)
        caz = section.z
        try:
            return 1000 * np.sum(abs(caz - y_coords))
        except:
            return 1e10
    elif isinstance(section, BezierAirfoil):
        section.generate_section(control_pts, nr_pts, delta_z_te)
        y_coords = airfoil_resplining2(ta.x, ta.z, section.x)
        caz = section.z
        try:
            return 1000 * np.sum(abs(caz - y_coords))
        except:
            return 1e10
    elif isinstance(section, BsplineAirfoil):
        order = args[0]['order']
        if len(control_pts) % 2 == 0:
            delta_z_te = 0
        else:
            delta_z_te = control_pts[-1]
            control_pts = np.delete(control_pts, -1)
        section.generate_section(control_pts, nr_pts, delta_z_te, order)

        y_coords = airfoil_resplining2(ta.x, ta.z, section.x)
        caz = section.z
        try:
            return 1000 * np.sum(abs(caz - y_coords))
        except:
            return 1e10
    elif isinstance(section, HicksHenneAirfoil):
        if len(control_pts) % 2 != 0:
            print('hicks henne does not have delta_Z_te yet')
        if 'base_airfoil' in args[0]:
            base_airfoil = args[0]['base_airfoil']
            section.generate_section(control_pts, nr_pts, base_airfoil=base_airfoil)
        else:
            section.generate_section(control_pts, nr_pts)
        y_coords = airfoil_resplining2(ta.x, ta.z, section.x)
        caz = section.z
        try:
            return 1000 * np.sum(abs(caz - y_coords))
        except:
            return 1e10
    elif isinstance(section, ParsecAirfoil):
        section.generate_section(control_pts, nr_pts)
        y_coords = airfoil_resplining2(ta.x, ta.z, section.x)
        caz = section.z
        try:
            return 1000 * np.sum(abs(caz - y_coords))
        except:
            return 1e10

def shape_gen(section, control_pts, nr_pts, order='order', base_airfoil='base_airfoil', delta_z_te ='delta_z_te'):
    if isinstance(section, CST):
        section.generate_section(control_pts, nr_pts, delta_z_te)
        return section
    elif isinstance(section, CSTmod):
        if len(control_pts) % 2 == 1:
            delta_z_te = 0
        else:
            delta_z_te = control_pts[-1]
            control_pts = np.delete(control_pts, -1)
        section.generate_section(control_pts, nr_pts, delta_z_te)
        return section
    elif isinstance(section, BezierAirfoil):
        if len(control_pts) % 2 == 0:
            delta_z_te = 0
        else:
            delta_z_te = control_pts[-1]
            control_pts = np.delete(control_pts, -1)
        section.generate_section(control_pts, nr_pts, delta_z_te)
        return section
    elif isinstance(section, BsplineAirfoil):
        if len(control_pts) % 2 == 0:
            delta_z_te = 0
        else:
            delta_z_te = control_pts[-1]
            control_pts = np.delete(control_pts, -1)
        section.generate_section(control_pts, nr_pts, delta_z_te, order)
        return section
    elif isinstance(section, HicksHenneAirfoil):
        if len(control_pts) % 2 != 0:
            print('hicks henne does not have delta_Z_te yet')
        section.generate_section(control_pts, nr_pts, base_airfoil=base_airfoil)
        return section
    elif isinstance(section, ParsecAirfoil):
        section.generate_section(control_pts, nr_pts)
        return section

def airfoil_resplining2(tax,taz, x_coords):
    x = tax
    y = taz
    # rescale airfoil to go from 0 to 1
    x_min = np.min(x)
    x_max = np.max(x)
    scale_factor = x_max - x_min
    x -= x_min
    x /= scale_factor
    y /= scale_factor

    # find min and split in top and bottom
    ind_min = np.argmin(x)
    x_top = x[0:ind_min+1]
    y_top = y[0:ind_min+1]
    x_bot = x[ind_min:]
    y_bot = y[ind_min:]

    # now let's spline
    ind_min2 = np.argmin(x_coords)
    x_upper = x_coords[0:ind_min2 + 1]
    x_lower = x_coords[ind_min2:]

    # f_top = interp1d(x_top, y_top, kind='cubic')
    # f_bot = interp1d(x_bot, y_bot, kind='cubic')
    f_top = interp1d(x_top, y_top, kind='cubic', fill_value='extrapolate', bounds_error=False)
    f_bot = interp1d(x_bot, y_bot, kind='cubic', fill_value='extrapolate', bounds_error=False)
    y_upper = f_top(x_upper)
    y_lower = f_bot(x_lower)

    y_basis = np.concatenate((y_upper, y_lower[1:]))
    return y_basis

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap