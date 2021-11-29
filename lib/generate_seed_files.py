from scipy.optimize import minimize
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.style as style
from lib.airfoil_parametrisation import AirfoilGeometry
from lib.airfoil_parametrisation import BsplineAirfoil, BezierAirfoil, CSTmod, CST, \
    HicksHenneAirfoil, ParsecAirfoil
from lib.airfoil_fitting import fit_airfoil
#from xfoil import XFoil
#from xfoil.model import Airfoil

# Colorblind-friendly colors
colors = [[0, 0, 0], '#0F95D7', [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255], [0, 158 / 255, 115 / 255],
          [230 / 255, 159 / 255, 0]]


nr_pts = 201
# to get list of all files in a folder
filelist = glob.glob("../airfoils_for_fitting_prop/*.dat")
savefolder = 'SeedAirfoils/'

# initialise to store
names = []
coefficients = []
plot_airfoil = True

#
# method = 'bspline' # 7 to 14 control points
# filenamesave = 'Bspline7Ctr_Seed_Airfoils.csv'
# bspline_nr_controlpts = 7
# b_upper = np.zeros(int(bspline_nr_controlpts - 2))
# b_lower = np.zeros(int(bspline_nr_controlpts - 2))
# order = bspline_nr_controlpts - 2



# method = 'bezier' # 10 to 24 in steps of 2
# bezierlength = 24
# b_upper = np.zeros(int(bezierlength/2))
# b_lower = np.zeros(int(bezierlength/2))
# order = bezierlength
# filenamesave = 'Bezier24_Seed_Airfoils.csv'

# method = 'bspline' # 7 to 14 in steps of 1
# filenamesave = 'BsplineCubic7Ctrl_Seed_Airfoils.csv'
# bspline_nr_controlpts = 7
# b_upper = np.zeros(int(bspline_nr_controlpts - 2))
# b_lower = np.zeros(int(bspline_nr_controlpts - 2))
# order = 3


# method = 'CST' # 3 to 10 in steps of 1
# filenamesave = 'CST10_Seed_Airfoils.csv'
# cst_order = 10
# b_upper = np.zeros(int(cst_order + 1))
# b_lower = np.zeros(int(cst_order + 1))
# order = cst_order

# method = 'CSTmod' # 3 to 10 in steps of 1
# filenamesave = 'CSTmod5_Seed_Airfoils.csv'
# cst_order = 5
# b_upper = np.zeros(int(cst_order + 1))
# b_lower = np.zeros(int(cst_order + 1))
# order = cst_order


method = 'hickshenne' # 5 to 12 in steps of 1
filenamesave = 'HicksHenne24_Seed_Airfoils.csv'
nr_bumps = 12
b_upper = np.zeros(int(nr_bumps))
b_lower = np.zeros(int(nr_bumps))
order = nr_bumps

# method = 'Parsec'
# filenamesave = 'Parsec11_Seed_Airfoils.csv'
# order = 3

delta_z_te = 0
b_leadingedge = 0


if method.lower() == 'parsec':
    # initial_airfoil = [0.0837961618809277, 0.390802420521655, 0.209726369541402, -2.74967855872026, 0,
    #                   -0.165756238952580, 0.00128490952503425, 0.489546487934853, 0.0221595476027768,
    #                   -0.121888487777930, -0.141588967719469]
    initial_airfoil = [0.0676119149001821, 0.386121148079336, 0.189085882001318, -1.49997939405297,
                      0.000974845343152555, -0.145891792699594, 0.00101845324128799, 0.434594956741146,
                      0.0332102903065701, -0.134141377766208, -0.235286192890846]

else:
    initial_airfoil = np.concatenate((b_upper, b_lower))


base_airfoil = 'naca2412.dat'
style.use('fivethirtyeight')

for cntr in range(len(filelist)):
    if method.lower() == 'parsec':
         initial_airfoil = [0.0676119149001821, 0.386121148079336, 0.189085882001318, -1.49997939405297,
                           0.000974845343152555, -0.145891792699594, 0.00101845324128799, 0.434594956741146,
                           0.0332102903065701, -0.134141377766208, -0.235286192890846]
    else:
        initial_airfoil = np.concatenate((b_upper, b_lower))

    airfoil_filename = filelist[cntr]
    head, tail = os.path.split(airfoil_filename)
    # print(filename[:-4])
    print('Airfoil', tail[:-4], 'is', cntr + 1, 'out of', len(filelist))
    airfoil_parametrisation = method
    parametrisation_order = order
    open_trailing_edge = False
    with_TE_thickness = False

    try:
        section, res = fit_airfoil(airfoil_filename, airfoil_parametrisation, parametrisation_order,
                              open_trailing_edge=open_trailing_edge, with_TE_thickness=with_TE_thickness,
                              plot_airfoil=plot_airfoil, verbose=True, nr_pts=nr_pts, delta_z_te=delta_z_te)

        # Check for cross-over
        section.calc_thickness_and_camber()
    except:
        print('something went wrong')
        section = AirfoilGeometry()
        section.thickness = -1.0*np.ones(2)
        res = None
    finally:
        if min(section.thickness) >= 0:
            names.append(tail[:-4])
            if len(coefficients) == 0:
                coefficients = res.x[:]
            else:
                coefficients = np.vstack([coefficients, res.x[:]])

import csv

filenamesave = savefolder + filenamesave

with open(filenamesave, 'a') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')
    for name, matrix_row in zip(names, coefficients):
        output_row = [name]
        output_row.extend(matrix_row)
        print(output_row)
        writer.writerow(output_row)

# test_row = [filename[9:-4]]
# test_row.extend(res.x)