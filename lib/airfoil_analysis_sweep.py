import numpy as np
import os, subprocess

from lib.utils import setup_solver
from lib import config, util
from lib.airfoil_parametrisation import AirfoilGeometry
from lib import flight
from lib.result import Result2 as Result
from lib.design import Design
from lib.cfd_lib import gmsh_util, construct2d_util, xfoil_util, mses_util
from cases.single_element_setup import SingleElementSetup
from lib.graphics import plot_geometry


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
# from xfoil import XFoil
# from xfoil.model import Airfoil

# plotutils has a function to truncate a colormap if needed
# from plotutils import truncate_colormap
# from matplotlib import cm
# import seaborn as sns
# cmap = cm.get_cmap('Blues', n_gen)
# cmap = truncate_colormap(cmap, 0.2, 0.8)

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
mygreen = [139/255, 195 / 255, 74/255]
mybrown = [153/255,79/255,0/255]
mydarkblue = [60/255, 105/255, 225/255]
mypurple = [0.4940, 0.1840, 0.5560]
myorange = [230/255, 97/255, 0/255]
mygray = [89 / 255, 89 / 255, 89 / 255]


colors = [myblack, myblue, myred, myyellow,
            mygreen, mybrown, mydarkblue, mypurple, myorange, mygray]
line_width = 2.75

"""
this is where I want Shahfiq to add an analysis functionality
"""


def airfoil_analysis_sweep(design, config, flight_condition, plot=False, **kwargs):

    airfoil = design.airfoil
    if config.settings.mesher == 'construct2d':
        delta_z_te = 0.0025
    else:
        delta_z_te = 0.0025

    # Generate the section and write airfoil to file
    sol_idx = 0
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists('./airfoils/'):
        os.makedirs('./airfoils/')
    rel_path = 'airfoils/' + config.case_name + '_' + str(sol_idx) + '.txt'
    abs_path = os.path.join(script_dir, rel_path)
    airfoil.write_coordinates_to_file(abs_path)

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
        cfd_eval = mses_util.mses_wrapper_sweep
    else:
        raise Exception('Solver not valid')



    # Evaluate airfoil performance
    try:
        if config.settings.solver.lower() in ['xfoil', 'xfoil_python']:
            current_result = cfd_eval(design, flight_condition, config.settings.solver, config.settings.n_core,
                                      identifier=sol_idx, use_python_xfoil=config.settings.use_python_xfoil, **kwargs)
        else:
            current_result = cfd_eval(design, flight_condition, config.settings.solver, config.settings.n_core,
                                      identifier=sol_idx, **kwargs)
    except Exception:
        current_result = None

    # Assign results
    if current_result:
        # Initialise result array
        result = Result(len(current_result.alpha))
        # Assign results
        result.alpha = current_result.alpha
        result.c_l = current_result.c_l
        result.c_d = current_result.c_d
        result.c_m = current_result.c_m
    else:
        print('Solver did not converge')

    print('lift coefficient =', result.c_l)
    print('drag coefficient =', result.c_d)
    print('----------------------------------------')

    if plot:
        plot_geometry(airfoil, **kwargs)

    subprocess.run(['rm -rf ' + abs_path], shell=True)

    return result

def write_results_to_file(filename, alpha, lift_coefficient, drag_coefficient, reynolds, mach):
    with open(filename, 'w') as f:
        string = os.path.split(airfoil_filename)[1][:-4] + '_Re_' + str(reynolds[0]) + '_Ma_' + str(round(100*mach[0])) \
               + '_xtr_' + str(round(100*xtr))
        f.write('%s\n' % string)
        for i in range(len(alpha)-1):
            f.write('%s %s %s\n' % (alpha[i], lift_coefficient[i], drag_coefficient[i]))
        f.write('%s %s %s' % (alpha[-1], lift_coefficient[-1], drag_coefficient[-1]))
    f.close()

def plot_robust_comparison():#
    # colors = [myblack, myblue, myred, myyellow, mygreen, mybrown, mydarkblue, mypurple, myorange, mygray]
    class data():
        def __init__(self):
            self.x = []
            self.z = []
            self.alpha = []
            self.cl = []
            self.cd = []
    det = data() # deterministic airfoil
    rob = data() # robust airfoil
    rob2= data()

    with open('../airfoils/utest.dat') as f:
        d = f.readlines()
        for i in d:
            line = i.rstrip().split("\t")
            try:
                det.x.append(float(line[0]))
                det.z.append(float(line[1]))
            except:
                pass
    with open('../airfoils/utest_unc.dat') as f:
        d = f.readlines()
        for i in d:
            line = i.rstrip().split("\t")
            try:
                rob.x.append(float(line[0]))
                rob.z.append(float(line[1]))
            except:
                pass
    with open('../airfoils/utest_unc2.dat') as f:
        d = f.readlines()
        for i in d:
            line = i.rstrip().split("\t")
            try:
                rob2.x.append(float(line[0]))
                rob2.z.append(float(line[1]))
            except:
                pass

    with open('../utest_unc_Re_200000_Ma_50_xtr_100') as fu:
        du = fu.readlines()
    with open('../utest_unc2_Re_200000_Ma_50_xtr_100') as fu:
        du2 = fu.readlines()
    with open('../utest_Re_200000_Ma_50_xtr_100') as f:
        d = f.readlines()
    for i in range(1,len(du2)):
        line = d[i].rstrip().split(" ")
        lineu= du[i].rstrip().split(" ")
        lineu2=du2[i].rstrip().split(" ")
        try:
            rob.alpha.append(float(lineu[0]))
            rob.cl.append(float(lineu[1]))
            rob.cd.append(float(lineu[2]))
            rob2.alpha.append(float(lineu2[0]))
            rob2.cl.append(float(lineu2[1]))
            rob2.cd.append(float(lineu2[2]))
            det.alpha.append(float(line[0]))
            det.cl.append(float(line[1]))
            det.cd.append(float(line[2]))
        except:
            pass
    # plot this shit
    fig1, ax1 = plt.subplots(1)
    ax1.plot(rob.x,rob.z,'red')
    ax1.plot(det.x,det.z,'blue')
    ax1.plot(rob2.x,rob2.z,'purple')
    ax1.axis('equal')
    fig2, ax2 = plt.subplots(1)
    ax2.plot(rob.cd, rob.cl, 'red')
    ax2.plot(det.cd, det.cl, 'blue')
    ax2.plot(rob2.cd,rob2.cl,'purple')

    plt.show()





if __name__ == '__main__':
    # airfoil_filename = '../airfoils_for_fitting_prop/MH117.dat'
    # airfoil_filename = '../airfoils/prop_0775r_thin_0.txt'
    # airfoil_filename = '../airfoils_for_fitting/e387.dat'
    # config.case_name = 'airfoil_analysis'

    # airfoil_filename    = '../airfoils/utest_unc.dat'
    # config.case_name    = 'utest_unc'
    airfoil_filename    = '../airfoils/utest_unc2.dat'
    config.case_name    = 'utest_unc2'
    # airfoil_filename    = '../airfoils/utest.dat'
    # config.case_name    = 'utest'

    plot_robust_comparison()
    quit()
    # ------------------------------------------------------------------------------------------------------------------
    # Set up solver
    solver_name = 'mses'  # options are 'xfoil','xfoil_python', 'mses',su2','openfoam','su2_gmsh','openfoam_gmsh'
    setup_solver(solver_name, config)

    # ------------------------------------------------------------------------------------------------------------------
    # Set up conditions and general settings
    nr_points = 151

    lift_coefficient = None
    alpha = np.linspace(-3, 8, 101)
    # alpha = [i for i in range(12)]
    alpha = [i*np.pi/180 for i in alpha]
    reynolds = [2.0e5]*np.ones(len(alpha))
    mach = [0.5]*np.ones(len(alpha))
    xtr = 1.0
    # ------------------------------------------------------------------------------------------------------------------
    # Convert into flight conditions
    # check if variables exist
    if not "lift_coefficient" in locals():
        if not "alpha" in locals():
            NameError('you need to define either alpha or lift coefficient')
        else:
            alpha = None
    elif not "alpha" in locals():
        alpha = None

    # actual conversion
    n_fc = len(reynolds)
    flight_condition = [flight.FlightCondition() for _ in range(n_fc)]
    for i, re in enumerate(reynolds):
        flight_condition[i].set(h=0.0, reynolds=re, mach=mach[i])
        if alpha is not None:
            flight_condition[i].alpha = alpha[i]
        elif lift_coefficient is not None:
            flight_condition[i].c_l = lift_coefficient[i]

    # ------------------------------------------------------------------------------------------------------------------
    # Load the geometry and create a finer geometry
    airfoil = AirfoilGeometry()
    airfoil.read_coordinates(airfoil_filename)
    # airfoil.refine_coordinates(nr_pts=nr_points)

    os.chdir('../')

    # ------------------------------------------------------------------------------------------------------------------
    # set up the loop
    setup = SingleElementSetup()

    design = Design(parametrisation_method='bezier',n_var=1,application_id=config.case_name,design_objectives=['max_lift'])
    design.airfoil = airfoil
    design.lift_coefficient = lift_coefficient
    design.alpha = alpha
    design.reynolds = reynolds
    design.mach = mach
    design.transition_location = [xtr, xtr]

    lift_coeff = []
    drag_coeff = []

    result = airfoil_analysis_sweep(design, config, flight_condition)

    lift_coeff = result.c_l
    drag_coeff = result.c_d
    alpha = result.alpha

    plt.rc('axes', linewidth=1.75)
    plt.figure('test')
    lift1 = plt.plot(alpha, lift_coeff, linestyle='solid', color=colors[0],linewidth=line_width,
                     fillstyle='full', marker='o', markerfacecolor=colors[0], markeredgecolor=colors[0],
                     markersize='5', markeredgewidth='2')
    plt.grid(b=True, which='major', linestyle='-', linewidth=0.75, color=colors[0])
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', linewidth=0.15, color=colors[0])
    plt.xlabel('Angle of Attack, $\\alpha$ [deg]')
    plt.ylabel('Lift Coefficient, $C_l$ [--]')
    plt.show()

    plt.rc('axes', linewidth=1.75)
    plt.figure('test')
    drag1 = plt.plot(drag_coeff*10**4, lift_coeff, linestyle='solid', color=colors[0],linewidth=line_width,
                     fillstyle='full', marker='o', markerfacecolor=colors[0], markeredgecolor=colors[0],
                     markersize='5', markeredgewidth='2')
    plt.grid(b=True, which='major', linestyle='-', linewidth=0.75, color=colors[0])
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', linewidth=0.15, color=colors[0])
    plt.xlabel('Drag Coefficient, $C_d \\times 10^{4}$ [--]')
    plt.ylabel('Lift Coefficient, $C_l$ [--]')
    plt.show()

    filename = os.path.split(airfoil_filename)[1][:-4] + '_Re_' + str(int(reynolds[0])) + '_Ma_' + \
               str(int(round(100 * mach[0])))+ '_xtr_' + str(int(round(100 * xtr)))


    write_results_to_file(filename, alpha, lift_coeff, drag_coeff, reynolds, mach)











