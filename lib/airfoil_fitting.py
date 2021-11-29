from scipy.optimize import minimize
import glob, os, csv
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.style as style
from lib.airfoil_parametrisation import BsplineAirfoil, BezierAirfoil, CSTmod, CST, HicksHenneAirfoil, ParsecAirfoil
from lib.airfoil_parametrisation import AirfoilGeometry
from lib.utils import shape_match, shape_gen, airfoil_resplining2
from scipy.interpolate import interp1d
from lib.cfd_lib.xfoil_util import Xfoil


def fit_airfoil(filename, parametrisation, order, open_trailing_edge=True, with_TE_thickness=True,
                plot_airfoil=False, verbose=False, **kwargs):
    """
    :param filename: location of the dat file for the airfoil that needs to be fitted
    :param parametrisation: type of parametrisation to be use
    :param order: order of the parametrisation
    :param open_trailing_edge: if True the trailing edge is not closed
    :param with_TE_thickness: it True trailing edge thickness is a design variable, if not true and open_trailing_edge
                                is true then a value for trailing_edge_thickness needs to be set
    :param plot_airfoil: if true the original points and the fitted shape will be plot
    :param verbose: set to true to display SLSQP output
    :return: TBD
    """

    # TODO: add trailing edge options for the other parametrisation methods (for now only fully tested for CST
    #  although it should work for some other methods)

    if 'nr_pts' not in kwargs:
        nr_pts = 251
    else:
        nr_pts = kwargs['nr_pts']
    if 'delta_z_te' not in kwargs:
        delta_z_te = 0
    else:
        delta_z_te = kwargs['delta_z_te']
    if 'base_airfoil' not in kwargs:
        base_airfoil = 'naca2412'
    else:
        base_airfoil = kwargs['base_airfoil']

    # load the coordinates
    original_airfoil = AirfoilGeometry()
    original_airfoil.read_coordinates(filename)

    # airfoil translation (to get leading edge at 0,0)
    original_airfoil.locate_leading_edge()
    if not original_airfoil.leading_edge_coordinates[0] == 0:
        original_airfoil.x -= original_airfoil.leading_edge_coordinates[0]

    if not original_airfoil.leading_edge_coordinates[1] == 0:
        original_airfoil.z -= original_airfoil.leading_edge_coordinates[1]

    # original_airfoil.locate_leading_edge()
    original_airfoil.write_coordinates_to_file(filename=filename)
    original_airfoil.read_coordinates(filename)

    # airfoil rotation through xfoil
    trailing_edge_average = (original_airfoil.z[0] + original_airfoil.z[-1])/2
    if not trailing_edge_average == 0:
        temp_airfoil = Xfoil(n_crit=9, transition_location=[1,1])
        temp_airfoil.rotate_airfoil(filename)
        original_airfoil.read_coordinates(filename)

    original_airfoil.refine_coordinates(nr_pts=nr_pts)

    # initialise the parametrisation for the fit
    section, control_points = initialise_section(parametrisation, order, with_TE_thickness, open_trailing_edge,
                                                 delta_z_te=delta_z_te, nr_pts=nr_pts)

    if verbose:
        print("---------------------------------------------------------")
        print("SLSQP started")
    t = time.time()
    try:
        if with_TE_thickness:
            additional = {'npts': nr_pts, 'section': section, 'target_airfoil': original_airfoil,
                          'order': order, 'base_airfoil': base_airfoil, 'with_TE': with_TE_thickness}
        else:
            additional = {'npts': nr_pts, 'section': section, 'target_airfoil': original_airfoil,
                          'order': order, 'delta_z_te': delta_z_te, 'with_TE': with_TE_thickness}
        if verbose:
            res = minimize(fun=shape_match, args=additional, x0=control_points, method='SLSQP', tol=1e-10,
                           options={'disp': True, 'maxiter': 10000})
            elapsed = time.time() - t
            print("Elapsed time: {:.2f} seconds".format(elapsed))
            print("Objective value: {:.4f}".format(res.fun))
            print("---------------------------------------------------------")
        else:
            res = minimize(fun=shape_match, args=additional, x0=control_points, method='SLSQP', tol=1e-10,
                           options={'disp': False, 'maxiter': 10000})

        matched_airfoil = matched_section(parametrisation, with_TE_thickness, res, nr_pts, delta_z_te)
        # Check for cross-over
        matched_airfoil.calc_thickness_and_camber()
    except:
        print('something went wrong')
        matched_airfoil.thickness = -1.0 * np.ones(2)
    finally:
        airfoil_filename = filename[:-4] + '_fit.dat'
        matched_airfoil.write_coordinates_to_file(airfoil_filename)
        if plot_airfoil:
            plot_fitted_airfoil(original_airfoil, matched_airfoil)


def plot_fitted_airfoil(original_coordinates, fitted_coordinates):
    """
    function to plot the fitted airfoil
    :param original_coordinates: coordinates read from the dat file
    :param fitted_coordinates: coordinates of the fitted shape
    :return: Nothing returned
    """

    # Colorblind-friendly colors
    colors = [[0, 0, 0], '#0F95D7', [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255], [0, 158 / 255, 115 / 255],
              [230 / 255, 159 / 255, 0]]
    style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(14, 5))
    plt.plot(original_coordinates.x, original_coordinates.z, fillstyle='none', marker='o', markerfacecolor='none',
             markeredgecolor=colors[0], linestyle='none', markersize='10', markeredgewidth='2')
    ax1 = plt.plot(fitted_coordinates.x, fitted_coordinates.z, '#0F95D7')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # ax1 = plt.plot(fitted_coordinates.x, fitted_coordinates.z, color=colors[1])
    # plt.legend()
    plt.show()


def initialise_section(parametrisation, order, with_TE_thickness, open_trailing_edge=False, nr_pts=251,
                       delta_z_te=0, **kwargs):
    """
    initialises and instance of the parametrisation scheme that is selected for the fitting of the airfoil
    :param parametrisation: type of parametrisation scheme to be used
    :param order: order of the parametrisation
    :param with_TE_thickness: boolean to indicate if trailing edge thickness is a parameter or not
    :param open_trailing_edge: boolean to indicate if the trailing edge is open or not
    :param nr_pts: number of points to be fitted to the airfoil
    :param delta_z_te: trailing edge thickness
    :param kwargs:
    :return:
    """
    if with_TE_thickness and not open_trailing_edge:
        print('if you want to fit trailing edge thickness you should allow open airfoils. I changed it for you')
        open_trailing_edge = True

    if parametrisation.lower() == 'cst':
        b_upper = np.zeros(int(order + 1))
        b_lower = np.zeros(int(order + 1))
        initial_airfoil = np.concatenate((b_upper, b_lower))
        section = CST()
        if with_TE_thickness:
            initial_airfoil = np.append(initial_airfoil, delta_z_te)
            section.generate_section(initial_airfoil[:-1:], npt=nr_pts, delta_z_te=initial_airfoil[-1])
        else:
            section.generate_section(initial_airfoil, npt=nr_pts, delta_z_te=delta_z_te)
    elif parametrisation.lower() == 'bspline':
        bspline_nr_controlpts = order + 2
        b_upper = np.zeros(int(bspline_nr_controlpts - 2))
        b_lower = np.zeros(int(bspline_nr_controlpts - 2))
        initial_airfoil = np.concatenate((b_upper, b_lower))
        section = BsplineAirfoil()
        if with_TE_thickness:
            initial_airfoil = np.append(initial_airfoil, delta_z_te)
            section.generate_section(initial_airfoil[:-1:], nr_pts, initial_airfoil[-1], order)
        else:
            section.generate_section(initial_airfoil, nr_pts, delta_z_te, order)
    elif parametrisation.lower() == 'cstmod':
        b_upper = np.zeros(int(order + 1))
        b_lower = np.zeros(int(order + 1))
        initial_airfoil = np.concatenate((b_upper, b_lower))
        if with_TE_thickness:
            initial_airfoil = np.append(initial_airfoil, delta_z_te)
        b_leadingedge = 0
        section = CSTmod()
        initial_airfoil = np.append(initial_airfoil, b_leadingedge)
        if open_trailing_edge:
            section.generate_section(initial_airfoil[:-1:], nr_pts, delta_z_te=initial_airfoil[-1])
        else:
            section.generate_section(initial_airfoil, nr_pts)

    elif parametrisation.lower() == 'parsec':
        initial_airfoil = [0.0676119149001821, 0.386121148079336, 0.189085882001318, -1.49997939405297,
                           delta_z_te, -0.145891792699594, 0.00101845324128799, 0.434594956741146,
                           0.0332102903065701, -0.134141377766208, -0.235286192890846]
        section = ParsecAirfoil()
        section.generate_section(initial_airfoil, nr_pts)

    elif parametrisation.lower() == 'bezier':
        b_upper = np.zeros(int(order / 2))
        b_lower = np.zeros(int(order / 2))
        initial_airfoil = np.concatenate((b_upper, b_lower))
        if with_TE_thickness:
            initial_airfoil = np.append(initial_airfoil, delta_z_te)
        section = BezierAirfoil()
        section.generate_section(initial_airfoil, nr_pts, delta_z_te)
    elif parametrisation.lower() == 'hickshenne':
        if 'base_airfoil' not in kwargs:
            base_airfoil = 'naca2412'
        b_upper = np.zeros(int(order))
        b_lower = np.zeros(int(order))
        initial_airfoil = np.concatenate((b_upper, b_lower))
        section = HicksHenneAirfoil()
        if with_TE_thickness:
            initial_airfoil = np.append(initial_airfoil, delta_z_te)
            print('not done yet - can not use hicks-henne with open trailing edge for now')
        else:
            section.generate_section(initial_airfoil, nr_pts)

    else:
        print(parametrisation.lower() + 'is not implemented yet')
    return section, initial_airfoil


def matched_section(parametrisation, with_TE_thickness, coefficients, nr_pts=251, delta_z_te=0, **kwargs):
    """
    generate the airfoil instance for the final matched section
    :param parametrisation: type of parametrisation scheme to be used
    :param with_TE_thickness: boolean to indicate if trailing edge thickness is a parameter or not
    :param coefficients: coefficients of the matched section
    :param nr_pts: number of points for the final airfoil
    :param delta_z_te: trailing edge thickness
    :return:
    """
    if parametrisation == 'bezier':
        matched_airfoil = BezierAirfoil()
        matched_airfoil = shape_gen(matched_airfoil, coefficients.x, nr_pts, delta_z_te)
        if with_TE_thickness:
            raise Exception("Sorry, cstmod does not support with_TE_thickness yet - use CST instead")
    elif parametrisation == 'cst':
        matched_airfoil = CST()
        if with_TE_thickness:
            matched_airfoil = shape_gen(matched_airfoil, coefficients.x[:-1:], nr_pts, delta_z_te=coefficients.x[-1])
        else:
            matched_airfoil = shape_gen(matched_airfoil, coefficients.x, nr_pts, delta_z_te=delta_z_te)
    elif parametrisation == 'cstmod':
        matched_airfoil = CSTmod()
        matched_airfoil = shape_gen(matched_airfoil, coefficients.x, nr_pts)
        if with_TE_thickness:
            raise Exception("Sorry, cstmod does not support with_TE_thickness yet - use CST instead")
    elif parametrisation == 'bspline':
        if 'order' not in kwargs:
            print('you did not specify the order for the bspline class. Set to cubic splines')
            order = 3
        matched_airfoil = BsplineAirfoil()
        matched_airfoil = shape_gen(matched_airfoil, coefficients.x, nr_pts, order=order)
        if with_TE_thickness:
            raise Exception("Sorry, cstmod does not support with_TE_thickness yet - use CST instead")
    elif parametrisation == 'hickshenne':
        if 'base_airfoil' not in kwargs:
            print('you did not set the base airfoil for Hicks Henne. Set to naca2412')
            base_airfoil = 'naca2412'
        matched_airfoil = HicksHenneAirfoil()
        matched_airfoil = shape_gen(matched_airfoil, coefficients.x, nr_pts, base_airfoil=base_airfoil)
        if with_TE_thickness:
            raise Exception("Sorry, cstmod does not support with_TE_thickness yet - use CST instead")
    elif parametrisation == 'parsec':
        matched_airfoil = ParsecAirfoil()
        matched_airfoil = shape_gen(matched_airfoil, coefficients.x, nr_pts)
        if with_TE_thickness:
            raise Exception("Sorry, cstmod does not support with_TE_thickness yet - use CST instead")
    else:
        print('not done yet')
    return matched_airfoil



if __name__ == '__main__':
    airfoil_filename = '../airfoils_for_testing/LA5055.dat'
    airfoil_parametrisation = 'cstmod'
    parametrisation_order = 4
    # ------------------------------------------------------------------------------------------------------------------
    # settings to fit an existing open TE airfoil
    open_trailing_edge = True  # set to true if you want the trailing edge to be open
    with_TE_thickness = False  # set to True if you want the trailing edge thickness to be fitted
    # ------------------------------------------------------------------------------------------------------------------
    # settings for construct2d for an airfoil with a close TE
    # open_trailing_edge = True # set to true if you want the trailing edge to be open
    # with_TE_thickness = False # set to True if you want the trailing edge thickness to be fitted
    # ------------------------------------------------------------------------------------------------------------------

    # to match trailing thickness of a given airfoil you would set with_TE_thickness to True
    plot_airfoil = True
    # possible additional arguments (as keyword arguments
    verbose = True  # default is false
    nr_pts = 251  # default is 251
    delta_z_te = 0.00254  # default is zero - set for construct2d

    # fit_airfoil(airfoil_filename, airfoil_parametrisation, parametrisation_order,
    #             open_trailing_edge=open_trailing_edge,
    #             with_TE_thickness=with_TE_thickness,
    #             plot_airfoil=plot_airfoil,
    #             verbose=True,
    #             nr_pts=nr_pts)

    fit_airfoil(airfoil_filename, airfoil_parametrisation, parametrisation_order,
                open_trailing_edge=open_trailing_edge,
                with_TE_thickness=with_TE_thickness,
                plot_airfoil=plot_airfoil,
                verbose=True,
                nr_pts=nr_pts,
                delta_z_te=delta_z_te)

    # fit_airfoil(airfoil_filename, airfoil_parametrisation, parametrisation_order,
    # open_trailing_edge=open_trailing_edge,
    # with_TE_thickness=with_TE_thickness,
    # plot_airfoil=plot_airfoil)
