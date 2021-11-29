import math

import numpy as np
import pyclipper
from scipy import spatial, interpolate
# TODO (Shahfiq) add documentation for the functions and check if this is still consistent with the rest (can we pass
#  conditions instance instead of individual rho, mu, ...)

def scale_calculations(rho, mu, chord_length, Re, y_plus, nr_of_points_bdly_y, growth_rate):
    # calculate freestream velocity based on Re
    U = Re*mu/rho/chord_length
    # calculate skin friction
    Cf = 0.058*Re**(-0.2)
    # calculate wall shear stress
    tau_w = 0.5*Cf*rho*U**2
    # calculate frictional velocity
    U_tau = math.sqrt(tau_w/rho)
    firstCellHeight = y_plus*mu/rho/U_tau
    # calculate the height of boundary layer
    factor = 0
    for n in range(nr_of_points_bdly_y):
        factor += growth_rate**n
    boundary_layer_scale = factor*firstCellHeight

    return boundary_layer_scale


def generate_scaled_coordinates(airfoil, airfoil_scaled, boundary_layer_scale, tolerance):
    # scaled with pyclipper
    airfoil_array = np.vstack((airfoil.x, airfoil.z)).T
    array_of_tuples = map(tuple, airfoil_array)
    airfoil_tuples = tuple(array_of_tuples)
    pco = pyclipper.PyclipperOffset()
    coordinates_scaled = pyclipper.scale_to_clipper(airfoil_tuples)
    pco.AddPath(coordinates_scaled, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    new_coordinates = pco.Execute(pyclipper.scale_to_clipper(boundary_layer_scale))
    new_coordinates_scaled = pyclipper.scale_from_clipper(new_coordinates)
    airfoil_scaled.x = np.array(new_coordinates_scaled[0])[:, 0]
    airfoil_scaled.z = np.array(new_coordinates_scaled[0])[:, 1]
    #  find trailing edge point and reorder scaled airfoil so that it goes TE -- LE -- TE
    x1 = airfoil.x[0]
    fraction = 0.05
    indexpnt = int(np.floor(fraction * len(airfoil.x)))
    x2 = (airfoil.x[indexpnt] + airfoil.x[-indexpnt]) / 2
    y1 = airfoil.z[0]
    y2 = (airfoil.z[indexpnt] + airfoil.z[-indexpnt]) / 2
    slope = (y2 - y1) / (x2 - x1)
    #  find the closest point on the new shape
    #  define the new point
    new_point_x = airfoil.x[0] + boundary_layer_scale
    new_point_z = airfoil.z[0] + boundary_layer_scale * slope
    # calculate the distances to work out the closest point
    #  create an array from the points
    A = np.asarray([airfoil_scaled.x, airfoil_scaled.z]).T
    pt = [new_point_x, new_point_z]
    distance, index = spatial.KDTree(A).query(pt)
    testx = airfoil_scaled.x[index:-1]
    testx = np.append(testx, airfoil_scaled.x[0:index])
    testz = airfoil_scaled.z[index:-1]
    testz = np.append(testz, airfoil_scaled.z[0:index])
    airfoil_scaled.x = testx
    airfoil_scaled.z = testz
    #  end of find trailing edge point and re-order
    # #  now respline with same number of points as original airfoil
    tck, u = interpolate.splprep([airfoil_scaled.x, airfoil_scaled.z], s=0, per=False)
    xi, yi = interpolate.splev(np.linspace(0, 1, len(airfoil.x) + 1), tck)
    airfoil_scaled.x = xi
    airfoil_scaled.z = yi
    # and fix the trailing edge
    if airfoil_scaled.z[0] == airfoil_scaled.z[-1]:
        airfoil_scaled.z[0] = airfoil_scaled.z[0] + tolerance
        airfoil_scaled.z[-1] = airfoil_scaled.z[-1] - tolerance

    return airfoil_scaled


def generate_boundary_layer_and_fan(airfoil,airfoil_scaled,boundary_layer_scale,with_fan,x_start,ratio,factor2):
    xlist = airfoil.x
    ylist = airfoil.z
    xout = []
    yout = []

    ind_le = np.argmin(airfoil.x)
    chord_length = abs(airfoil.x[-1] - airfoil.x[ind_le])
    for cntr in range(1, len(airfoil.x) - 1):
        # x1 = xlist[cntr+1]
        # x2 = xlist[cntr]
        # x3 = xlist[cntr+2]
        # y1 = ylist[cntr+1]
        # y2 = ylist[cntr]
        # y3 = ylist[cntr+2]
        x1 = xlist[cntr]
        x2 = xlist[cntr-1]
        x3 = xlist[cntr+1]
        y1 = ylist[cntr]
        y2 = ylist[cntr-1]
        y3 = ylist[cntr+1]
        dx1 = x1-x2
        dy1 = y1-y2
        length_1 = math.sqrt(dx1**2+dy1**2)
        vector1 = [dx1, 0, dy1]
        # try:
        #     vector1_nom = [x / length_1 for x in vector1]
        # except RuntimeWarning:
        #     vector1_nom = vector1_nom_minus1
        # finally:
        #     vector1_nom_minus1 = vector1_nom
        if length_1 != 0:
            vector1_nom = [x / length_1 for x in vector1]
            vector1_nom_minus1 = vector1_nom
        else:
            vector1_nom = vector1_nom_minus1

        # vector1_nom = [x / length_1 for x in vector1]
        dx2 = x3-x1
        dy2 = y3-y1
        length_2 = math.sqrt(dx2**2+dy2**2)
        vector2 = [dx2, 0, dy2]
        # try:
        #     vector2_nom = [x / length_2 for x in vector2]
        # except RuntimeWarning:
        #     vector2_nom = vector2_nom_minus1
        # finally:
        #     vector2_nom_minus1 = vector2_nom
        if length_2 != 0:
            vector2_nom = [x / length_2 for x in vector2]
            vector2_nom_minus1 = vector2_nom
        else:
            vector2_nom = vector2_nom_minus1
        # vector2_nom = [x / length_2 for x in vector2]
        vector_ref = [0, -1, 0]
        perpen1 = np.cross(vector1_nom, vector_ref)
        perpen2 = np.cross(vector2_nom, vector_ref)
        perpen_final = perpen2 + perpen1
        perpen_final_nom = [x / math.sqrt(2) for x in perpen_final]
        # xnew = xlist[cntr + 1] + boundary_layer_scale * perpen_final_nom[0]
        # ynew = ylist[cntr + 1] + boundary_layer_scale * perpen_final_nom[2]
        xnew = xlist[cntr] + boundary_layer_scale * perpen_final_nom[0]
        ynew = ylist[cntr] + boundary_layer_scale * perpen_final_nom[2]

        # calculate current chord location x/c
        chord_ratio = abs(airfoil.x[cntr] - airfoil.x[ind_le])
        if chord_length > x_start:
            factor = 1 + (ratio-1)/(1-x_start)*(chord_ratio-x_start)
        else:
            factor = 1

        xout.append(float(xnew))
        yout.append(float(ynew))
    xout = np.asarray(xout)
    yout = np.asarray(yout)

    airfoil_scaled.x[1:-2] = xout
    airfoil_scaled.z[1:-2] = yout

    x1 = xlist[1]
    x2 = xlist[0]
    y1 = ylist[1]
    y2 = ylist[0]
    dx1 = x1 - x2
    dy1 = y1 - y2
    xnew1 = airfoil_scaled.x[1] - dx1
    znew1 = airfoil_scaled.z[1] - dy1

    airfoil_scaled.x[0] = xnew1
    airfoil_scaled.z[0] = znew1

    # first point after adjacent (lower surface)
    x1 = xlist[-2]
    x2 = xlist[-3]
    y1 = ylist[-2]
    y2 = ylist[-3]
    dx1 = x1 - x2
    dy1 = y1 - y2
    xnew2 = airfoil_scaled.x[-3] + dx1
    znew2 = airfoil_scaled.z[-3] + dy1

    airfoil_scaled.x[-2] = xnew2
    airfoil_scaled.z[-2] = znew2

    if with_fan:
        # bottom surface
        dx_lower = airfoil_scaled.x[-2] - airfoil_scaled.x[-3]
        dy_lower = airfoil_scaled.z[-2] - airfoil_scaled.z[-3]
        if dx_lower == 0:
            angle_box_lower = np.pi / 2
        else:
            angle_box_lower = np.arctan(abs(dy_lower/dx_lower))
        xnew_box_lower = airfoil_scaled.x[-2] + factor2*boundary_layer_scale * np.cos(angle_box_lower)
        ynew_box_lower = airfoil_scaled.z[-2] - factor2*boundary_layer_scale * np.sin(angle_box_lower)
        airfoil_scaled.x[-1] = xnew_box_lower
        airfoil_scaled.z[-1] = ynew_box_lower
        # top surface
        dx_upper = airfoil_scaled.x[0] - airfoil_scaled.x[1]
        dy_upper = airfoil_scaled.z[0] - airfoil_scaled.z[1]
        if dx_upper == 0:
            angle_box_upper = np.pi / 2
        else:
            angle_box_upper = np.arctan(abs(dy_upper / dx_upper))
        xnew_box_upper = airfoil_scaled.x[0] + factor2*boundary_layer_scale * np.cos(angle_box_upper)
        ynew_box_upper = airfoil_scaled.z[0] - factor2*boundary_layer_scale * np.sin(angle_box_upper)
        airfoil_scaled.x = np.append(airfoil_scaled.x, xnew_box_upper)
        airfoil_scaled.z = np.append(airfoil_scaled.z, ynew_box_upper)

        length = math.sqrt(dx_lower**2 + dy_lower**2)

    else:
        airfoil_scaled.x = airfoil_scaled.x[:-1]
        airfoil_scaled.z = airfoil_scaled.z[:-1]
        length = 0

    return airfoil, airfoil_scaled, length


def calculate_lc(nr_of_points_bdly_y, growth_rate, boundary_layer_scale):
    factor = 0
    for n in range(nr_of_points_bdly_y):
        factor += growth_rate ** n
    first_cell_height = boundary_layer_scale / factor
    lc = growth_rate ** (nr_of_points_bdly_y - 1) * first_cell_height

    return lc