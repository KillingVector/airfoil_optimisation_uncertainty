import numpy as np

from lib import airfoil_parametrisation


def generate_airfoil_instances(parametrisation):
    if parametrisation.lower() == 'bspline':
        airfoil = airfoil_parametrisation.BsplineAirfoil()
        airfoil_scaled = airfoil_parametrisation.BsplineAirfoil()
    elif parametrisation.lower() == 'bezier':
        airfoil = airfoil_parametrisation.BezierAirfoil()
        airfoil_scaled = airfoil_parametrisation.BezierAirfoil()
    elif parametrisation.lower() == 'cst':
        airfoil = airfoil_parametrisation.CST()
        airfoil_scaled = airfoil_parametrisation.CST()
    elif parametrisation.lower() == 'cstmod':
        airfoil = airfoil_parametrisation.CSTmod()
        airfoil_scaled = airfoil_parametrisation.CSTmod()
    elif parametrisation.lower() == 'hickshenne':
        airfoil = airfoil_parametrisation.HicksHenneAirfoil()
        airfoil_scaled = airfoil_parametrisation.HicksHenneAirfoil()
    else:
        raise Exception('not done yet')

    return airfoil, airfoil_scaled


def generate_spacing_vector(nr_pts, multiplier=101):

    # lin spacing for the trailing edge
    nr_pts *= multiplier
    x1 = np.linspace(0.5, 1, int(np.floor(nr_pts / 6) + 1))
    x1 = x1[1:-1:multiplier]
    if x1[-1] < 1:
        x1 = x1 = np.append(x1,[1])
    nr_pts /= multiplier

    # cosine spacing for the rest of the airfoil
    theta1 = np.linspace(np.pi / 2, 0.99 * np.pi, int(nr_pts / 3) - 1)
    # theta2 = np.linspace(0.99 * np.pi, 1.01 * np.pi, 5)
    theta2 = np.linspace(0.99 * np.pi, 1.01 * np.pi, int(np.floor(5 / 251 * nr_pts)))
    theta3 = np.linspace(1.01 * np.pi, 3 / 2 * np.pi, int(nr_pts / 3) - 1)
    theta = np.concatenate((theta1, theta2[1:-1], theta3))
    x2 = 0.5 * (np.cos(theta) + 1.0)
    xvec = np.concatenate((x1[::-1], x2[1:-1:], x1))

    return xvec

