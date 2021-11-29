import sys

import gmsh
import numpy as np

from lib.cfd_lib import airfoil_util, boundary_layer
from lib.cfd_lib.util import su2_wrapper, openfoam_wrapper
import lib.solver_settings as solver_settings

def generate_farfield(domain_size, lc_ff, circle_cntr):
    #  create 4 points
    gmsh.model.geo.addPoint(-domain_size / 2, 0, 0, lc_ff, 10001)
    gmsh.model.geo.addPoint(0, domain_size / 2, 0, lc_ff, 10002)
    gmsh.model.geo.addPoint(domain_size / 2, 0, 0, lc_ff, 10003)
    gmsh.model.geo.addPoint(0, -domain_size / 2, 0, lc_ff, 10004)
    # and center point
    gmsh.model.geo.addPoint(0, 0, 0, lc_ff, 10005)
    # add circle segments
    gmsh.model.geo.addCircleArc(10001, 10005, 10002, circle_cntr)
    gmsh.model.geo.addCircleArc(10002, 10005, 10003, circle_cntr + 1)
    gmsh.model.geo.addCircleArc(10003, 10005, 10004, circle_cntr + 2)
    gmsh.model.geo.addCircleArc(10004, 10005, 10001, circle_cntr + 3)
    circle_line_list = [circle_cntr, circle_cntr + 1, circle_cntr + 2, circle_cntr + 3]
    gmsh.model.geo.addCurveLoop(circle_line_list, 2)
    return circle_line_list


def generate_airfoil(airfoil, cntr_shift, airfoil_shift, nr_of_points_bdly_x, lc, with_fan):
    for cntr in range(len(airfoil.x)):
        gmsh.model.geo.addPoint(airfoil.x[cntr], airfoil.z[cntr],
                                0, lc, cntr + cntr_shift + airfoil_shift + 1)
    for cntrl in range(len(airfoil.x) - 1):
        gmsh.model.geo.addLine(cntrl + 1 + cntr_shift + airfoil_shift, cntrl + 2 + cntr_shift + airfoil_shift,
                               cntrl + 1 + cntr_shift + airfoil_shift)
        gmsh.model.geo.mesh.setTransfiniteCurve(cntrl + 1 + cntr_shift + airfoil_shift, nr_of_points_bdly_x,
                                                "Progression", 1)
    # add line to close trailing edge
    gmsh.model.geo.addLine(len(airfoil.x) + cntr_shift + airfoil_shift, 1 + cntr_shift + airfoil_shift,
                           len(airfoil.x) + cntr_shift + airfoil_shift)
    curve_list = list(range(1, len(airfoil.x) + 1))
    new_list = [x + cntr_shift + airfoil_shift for x in curve_list]
    if with_fan:
        gmsh.model.geo.addCurveLoop(new_list, cntr_shift + airfoil_shift + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(cntr_shift + airfoil_shift + len(airfoil.x), nr_of_points_bdly_x,
                                                "Progression", 1)
    return new_list


def generate_fan_scaled_airfoil(airfoil, cntr_shift, airfoil_shift, nr_of_points_bdly_x, nr_of_points_bdly_y, lc, length, boundary_layer, factor):

    for cntr in range(len(airfoil.x)):
        gmsh.model.geo.addPoint(airfoil.x[cntr], airfoil.z[cntr], 0, lc, cntr + cntr_shift + airfoil_shift + 1)
    fan_x_pts = round(factor*boundary_layer/length) + 1
    # if fan_x_pts < 17:
    #     fan_x_pts = 17
    for cntrl in range(len(airfoil.x) - 1):
        gmsh.model.geo.addLine(cntrl + 1 + cntr_shift + airfoil_shift, cntrl + 2 + cntr_shift + airfoil_shift,
                               cntrl + 1 + cntr_shift + airfoil_shift)
        if cntrl + 1 == len(airfoil.x) - 2:
            gmsh.model.geo.mesh.setTransfiniteCurve(cntrl + 1 + cntr_shift + airfoil_shift, fan_x_pts, "Progression", 1)
        elif cntrl + 1 == len(airfoil.x) - 1:
            gmsh.model.geo.mesh.setTransfiniteCurve(cntrl + 1 + cntr_shift + airfoil_shift, 2 * nr_of_points_bdly_y,
                                                    "Progression", 1)
        else:
            gmsh.model.geo.mesh.setTransfiniteCurve(cntrl + 1 + cntr_shift + airfoil_shift, nr_of_points_bdly_x,
                                                    "Progression", 1)
    # add line to close trailing edge
    gmsh.model.geo.addLine(len(airfoil.x) + cntr_shift + airfoil_shift, 1 + cntr_shift + airfoil_shift,
                           len(airfoil.x) + cntr_shift + airfoil_shift)
    gmsh.model.geo.mesh.setTransfiniteCurve(cntr_shift + len(airfoil.x) + airfoil_shift, fan_x_pts,
                                            "Progression", 1)
    curve_list = list(range(1, len(airfoil.x) + 1))
    new_list = [x + cntr_shift + airfoil_shift for x in curve_list]
    gmsh.model.geo.addCurveLoop(new_list, cntr_shift + 1 + airfoil_shift)
    return new_list


def generate_nofan_scaled_airfoil(airfoil, cntr_shift, airfoil_shift, nr_of_points_bdly_x, lc):

    for cntr in range(len(airfoil.x)):
        gmsh.model.geo.addPoint(airfoil.x[cntr], airfoil.z[cntr], 0, lc, cntr + cntr_shift + airfoil_shift + 1)
    for cntrl in range(len(airfoil.x)-1):
        gmsh.model.geo.addLine(cntrl + 1 + cntr_shift + airfoil_shift, cntrl + 2 + cntr_shift + airfoil_shift,
                               cntrl + 1 + cntr_shift + airfoil_shift)
        gmsh.model.geo.mesh.setTransfiniteCurve(cntrl + 1 + cntr_shift + airfoil_shift, nr_of_points_bdly_x,
                                                "Progression", 1)
    curve_list = list(range(1, len(airfoil.x)))
    new_list = [x + cntr_shift + airfoil_shift for x in curve_list]
    return new_list


def generate_line_connecting_airfoil_and_scaled_airfoil(BL_tag, scaled_shift, airfoil_shift, nr_of_points_bdly_y, growth_rate, with_fan, N):
    if with_fan:
        for cntr in range(N - 2):
            indxline = BL_tag + cntr + airfoil_shift
            gmsh.model.geo.addLine(1 + cntr + airfoil_shift, scaled_shift + 1 + cntr + airfoil_shift, indxline)
            gmsh.model.geo.mesh.setTransfiniteCurve(indxline, nr_of_points_bdly_y, "Progression", growth_rate)
            transfinite_list = list(range(0, N-1))
    else:
        for cntr in range(N):
            indxline = BL_tag + cntr + airfoil_shift
            gmsh.model.geo.addLine(1 + cntr + airfoil_shift, scaled_shift + 1 + cntr + airfoil_shift, indxline)
            gmsh.model.geo.mesh.setTransfiniteCurve(indxline, nr_of_points_bdly_y, "Progression", growth_rate)
            transfinite_list = list(range(0, N))
    transfinite_line_list = [x + BL_tag + airfoil_shift for x in transfinite_list]
    return transfinite_line_list


def curveloop_no_fan(airfoil_line_list, transfinite_line_list, scaled_airfoil_line_list, airfoil_shift):
    last_line = airfoil_line_list[-1]
    airfoil_list = airfoil_line_list
    airfoil_line_list = np.append(airfoil_line_list,transfinite_line_list[0])
    airfoil_line_list = np.append(airfoil_line_list,scaled_airfoil_line_list)
    airfoil_line_list = np.append(airfoil_line_list,-1*transfinite_line_list[-1])
    airfoil_line_list = np.append(airfoil_line_list,last_line)
    gmsh.model.geo.addCurveLoop(airfoil_line_list, 1 +  airfoil_shift)
    return airfoil_list


def generate_transfinite_mesh(BL_curve_cntr, BL_surf_cntr, cntr_shift, airfoil_shift, N, N_airfoil, with_fan, BL_tag, element=1):
    if with_fan:
        for cntr in range(N - 3):
            idxcurveloop = BL_curve_cntr + cntr + airfoil_shift + 1
            gmsh.model.geo.addCurveLoop([cntr + 1 + airfoil_shift, BL_tag + cntr + 1 + airfoil_shift,
                                         -(cntr + 1 + cntr_shift + airfoil_shift), -(BL_tag + cntr + airfoil_shift)],
                                        idxcurveloop)
            gmsh.model.geo.addPlaneSurface([idxcurveloop], BL_surf_cntr + cntr + 1 + airfoil_shift)
            gmsh.model.geo.mesh.setTransfiniteSurface(BL_surf_cntr + cntr +  airfoil_shift + 1, "Left",
                                                      [cntr + 2 + airfoil_shift, cntr + 1 + cntr_shift + airfoil_shift,
                                                       cntr + 2 + cntr_shift + airfoil_shift,
                                                       cntr + 1 + airfoil_shift])
            gmsh.model.geo.mesh.setRecombine(2, BL_surf_cntr + cntr + 1 + airfoil_shift)
        gmsh.model.geo.addLine(cntr_shift + 1 + airfoil_shift, cntr_shift + N + airfoil_shift, int(1.2882e8) +
                               airfoil_shift)
        cntr += 1
        idxcurveloop = BL_curve_cntr + cntr + 1 + airfoil_shift
        gmsh.model.geo.addCurveLoop(
            [cntr_shift + cntr + 1 + airfoil_shift, cntr_shift + cntr + 2 + airfoil_shift,
             cntr_shift + cntr + 3 + airfoil_shift, int(1.2882e8) + airfoil_shift],
            idxcurveloop + 1)
        gmsh.model.geo.addPlaneSurface([idxcurveloop + 1], BL_surf_cntr + cntr + 2 + airfoil_shift)
        gmsh.model.geo.remove([(2, BL_surf_cntr + cntr + 2 + airfoil_shift), (1, int(1.2882e8) + airfoil_shift)])
        gmsh.model.geo.addCurveLoop(
            [cntr_shift + cntr + 1 + airfoil_shift, cntr_shift + cntr + 2 + airfoil_shift,
             cntr_shift + cntr + 3 + airfoil_shift, -BL_tag - airfoil_shift, -N_airfoil - airfoil_shift,
             BL_tag + N_airfoil - 1 + airfoil_shift], idxcurveloop)
        gmsh.model.geo.addPlaneSurface([idxcurveloop], BL_surf_cntr + cntr + 1 + airfoil_shift)
        gmsh.model.geo.mesh.setTransfiniteSurface(BL_surf_cntr + cntr + 1 + airfoil_shift, "Left",
                                                  [cntr_shift + N_airfoil + airfoil_shift,
                                                   cntr_shift + N_airfoil + 1 + airfoil_shift,
                                                   cntr_shift + N_airfoil + 2 + airfoil_shift,
                                                   cntr_shift + 1 + airfoil_shift])
        gmsh.model.geo.mesh.setRecombine(2, BL_surf_cntr + cntr + 1 + airfoil_shift)
        BL_surface_cntr = list(range(BL_surf_cntr + 1 + airfoil_shift, BL_surf_cntr + cntr + 1 + airfoil_shift + 1))
    else:
        for cntr in range(N - 1):
            idxcurveloop = BL_curve_cntr + cntr + 1 + airfoil_shift
            gmsh.model.geo.addCurveLoop([cntr + 1 + airfoil_shift, BL_tag + cntr + 1 + airfoil_shift,
                                         -(cntr + 1 + cntr_shift + airfoil_shift), -(BL_tag + cntr + airfoil_shift)],
                                        idxcurveloop)
            gmsh.model.geo.addPlaneSurface([idxcurveloop], BL_surf_cntr + cntr + 1 + airfoil_shift)
            gmsh.model.geo.mesh.setTransfiniteSurface(BL_surf_cntr + cntr + 1 + airfoil_shift, "Left",
                                                      [cntr + 2 + airfoil_shift, cntr + 1 + cntr_shift + airfoil_shift,
                                                       cntr + 2 + cntr_shift + airfoil_shift,
                                                       cntr + 1 + airfoil_shift])
            gmsh.model.geo.mesh.setRecombine(2, BL_surf_cntr + cntr + 1 + airfoil_shift)
        BL_surface_cntr = list(range(BL_surf_cntr + 1 + airfoil_shift, BL_surf_cntr + N + airfoil_shift + 1))
    return BL_surface_cntr


def addpoint_behind_fan(airfoil_scaled, points_tag, airfoil_shift, lc, plane_cntr):
    dx = airfoil_scaled.x[-1] - airfoil_scaled.x[-2]
    dz = airfoil_scaled.z[-1] - airfoil_scaled.z[-2]
    if dx != 0:
        fan_gradient = dz/dx
        points_x = np.linspace(airfoil_scaled.x[-2], airfoil_scaled.x[-1], 20, endpoint=True)
        for cntr in range(len(points_x)):
            points_y = fan_gradient * (points_x[cntr] - airfoil_scaled.x[-1]) + airfoil_scaled.z[-1]
            gmsh.model.geo.addPoint(1.00005 * points_x[cntr], points_y, 0, 2 * lc, points_tag + cntr + 1 + airfoil_shift)
    else:
        points_x = airfoil_scaled.x[-1]*np.ones(20)
        for cntr in range(len(points_x)):
            points_y = cntr / 19 * dz + airfoil_scaled.z[-2]
            gmsh.model.geo.addPoint(1.00005 * points_x[cntr], points_y, 0, 2 * lc, points_tag + cntr + 1 + airfoil_shift)
    gmsh.model.geo.synchronize()
    for cntr in range(len(points_x)):
        gmsh.model.mesh.embed(0, [points_tag + cntr + 1 + airfoil_shift], 2, plane_cntr + 3)


def add_physical_surface(airfoil_list, circle_line_list, surface_cntr):
    gmsh.model.geo.synchronize()
    ps_airfoil = gmsh.model.addPhysicalGroup(1, airfoil_list)
    gmsh.model.setPhysicalName(1, ps_airfoil, "airfoil")
    ps_farfield = gmsh.model.addPhysicalGroup(1, circle_line_list)
    gmsh.model.setPhysicalName(1, ps_farfield, "farfield")
    ps_field = gmsh.model.addPhysicalGroup(2, surface_cntr)
    gmsh.model.setPhysicalName(2, ps_field, "field")


def gmsh_add_physical_surface_openfoam(plane_cntr, with_fan, BL_surf_cntr, n1, n2, n3, n_scaled1, n_scaled2, n_scaled3,
                                       airfoil_shift):
    e1 = gmsh.model.geo.extrude([(2, plane_cntr + 3)], 0, 0, 1.0, [1], recombine=True)
    gmsh.model.geo.synchronize()
    farfield_tag = int(1e4)
    farfield_e = []
    for cntr in range(2, 6):
        farfield_e.append(float(e1[cntr][1]))
    farfield_e2 = np.asarray(farfield_e)
    gmsh.model.addPhysicalGroup(2, farfield_e2, farfield_tag)
    gmsh.model.setPhysicalName(2, farfield_tag, name="farfield")
    airfoil_tag = int(1e5)
    airfoil_e = []
    if with_fan:
        for cntr in range(6, 6 + n1+n2 + 5):
            airfoil_e.append(float(e1[cntr][1]))
        airfoil_e2 = np.asarray(airfoil_e)
        gmsh.model.addPhysicalGroup(2, airfoil_e2, airfoil_tag)
        gmsh.model.setPhysicalName(2, airfoil_tag, name="airfoils")
    elif not with_fan:
        if n2 == 0 and n3 == 0:
            for cntr in range(6, 6 + 2 * n1 + 2):
                airfoil_e.append(float(e1[cntr][1]))
            airfoil_e2 = np.asarray(airfoil_e)
            gmsh.model.addPhysicalGroup(2, airfoil_e2, airfoil_tag)
            gmsh.model.setPhysicalName(2, airfoil_tag, name="airfoils")
        elif n2 != 0 and n3 == 0:
            for cntr in range(6, 6 + 2 * (n1 + n2) + 4):
                airfoil_e.append(float(e1[cntr][1]))
            airfoil_e2 = np.asarray(airfoil_e)
            gmsh.model.addPhysicalGroup(2, airfoil_e2, airfoil_tag)
            gmsh.model.setPhysicalName(2, airfoil_tag, name="airfoils")
        else:
            for cntr in range(6, 6 + 2 * (n1 + n2 + n3) + 6):
                airfoil_e.append(float(e1[cntr][1]))
            airfoil_e2 = np.asarray(airfoil_e)
            gmsh.model.addPhysicalGroup(2, airfoil_e2, airfoil_tag)
            gmsh.model.setPhysicalName(2, airfoil_tag, name="airfoils")
    field_grid = [e1[1][1]]
    if with_fan:
        for cntr in range(n_scaled1 - 2):
            e2 = gmsh.model.geo.extrude([(2, BL_surf_cntr + cntr + 1)], 0, 0, 1.0, [1], recombine=True)
            gmsh.model.geo.synchronize()
            field_grid.append(e2[1][1])
        if n_scaled2 != 0:
            for cntr in range(n_scaled2 - 2):
                e2 = gmsh.model.geo.extrude([(2, BL_surf_cntr + cntr + 1 + airfoil_shift)], 0, 0, 1.0, [1], recombine=True)
                gmsh.model.geo.synchronize()
                field_grid.append(e2[1][1])
        if n_scaled3 != 0:
            for cntr in range(n_scaled3 - 2):
                e2 = gmsh.model.geo.extrude([(2, BL_surf_cntr + cntr + 1 + 2*airfoil_shift)], 0, 0, 1.0, [1], recombine=True)
                gmsh.model.geo.synchronize()
                field_grid.append(e2[1][1])
        gmsh.model.addPhysicalGroup(3, field_grid, int(2.981e9))
        gmsh.model.setPhysicalName(3, int(2.981e9), "internal")
    else:
        for cntr in range(n_scaled1 - 1):
            e2 = gmsh.model.geo.extrude([(2, BL_surf_cntr + cntr + 1)], 0, 0, 1.0, [1], recombine=True)
            gmsh.model.geo.synchronize()
            field_grid.append(e2[1][1])
        if n_scaled2 != 0:
            for cntr in range(n_scaled2 - 1):
                e2 = gmsh.model.geo.extrude([(2, BL_surf_cntr + cntr + 1 + airfoil_shift)], 0, 0, 1.0, [1], recombine=True)
                gmsh.model.geo.synchronize()
                field_grid.append(e2[1][1])
        if n_scaled3 != 0:
            for cntr in range(n_scaled3 - 1):
                e2 = gmsh.model.geo.extrude([(2, BL_surf_cntr + cntr + 1 + 2*airfoil_shift)], 0, 0, 1.0, [1], recombine=True)
                gmsh.model.geo.synchronize()
                field_grid.append(e2[1][1])
        gmsh.model.addPhysicalGroup(3, field_grid, int(2.981e9))
        gmsh.model.setPhysicalName(3, int(2.981e9), "internal")


def run_single_element(design, flight_condition, solver, num_core, **kwargs):

    rho = flight_condition.ambient.rho
    mu = flight_condition.ambient.viscosity.mu
    Re = flight_condition.reynolds
    AoA = flight_condition.alpha

    domain_size = 40    # times the chord length
    with_fan = False
    tolerance = 1e-7
    fan_length = 1  # this will be multiples of boundary layers scale factor
    nr_of_points_bdly_x = 1
    nr_of_points_bdly_y = 50
    growth_rate = 1.1
    y_plus = 0.5

    # settings for trailing edge boundary layers mesh
    x_start = 0.5
    BL_ratio = 1

    # gmsh tag
    # set off-set values for various parts of the mesh
    circle_cntr = int(1e5+2e4)
    plane_cntr = int(9e6)
    cntr_shift = int(2.5e5)
    BL_tag = 123450
    BL_curve_cntr = 1234560
    BL_surf_cntr = 12345670
    point_tag = 9999999

    # create some empty instances for the airfoil and the scaled airfoil (used to generate the boundary layer)
    airfoil, airfoil_scaled = airfoil_util.generate_airfoil_instances(design.parametrisation_method)

    # Assign airfoil coordinates
    chord_length = 1.0
    x_spacing = airfoil_util.generate_spacing_vector(design.n_pts)
    airfoil.generate_section(design.shape_variables, design.n_pts, xvec=x_spacing)

    # Scaling coordinates
    airfoil.x *= chord_length
    airfoil.z *= chord_length

    # tolerance check
    if airfoil.z[0] == airfoil.z[-1]:
        airfoil.z[0] += tolerance
        airfoil.z[-1] -= tolerance

    # calculate boundary layer scale
    boundary_layer_scale = boundary_layer.scale_calculations(rho, mu, chord_length, Re/chord_length,
                                                             y_plus, nr_of_points_bdly_y, growth_rate)

    #  generate the scaled airfoil to define the boundary layer
    airfoil_scaled = boundary_layer.generate_scaled_coordinates(airfoil, airfoil_scaled, boundary_layer_scale,
                                                                tolerance)

    #  generate the boundary layer and fan points
    airfoil, airfoil_scaled, difference_length = boundary_layer.generate_boundary_layer_and_fan(airfoil, airfoil_scaled,
                                                                                 boundary_layer_scale, with_fan,
                                                                                 x_start, BL_ratio, fan_length)
    # Generate mesh with gmsh
    gmsh.initialize()
    gmsh.model.add("name")
    gmsh.option.setNumber("General.Terminal", 0)

    lc_ff = 5e-1/30*domain_size

    lc = boundary_layer.calculate_lc(nr_of_points_bdly_y, growth_rate, boundary_layer_scale)

    #  Generate the farfield
    circle_line_list = generate_farfield(domain_size, lc_ff, circle_cntr)

    #  Generate the main airfoil
    airfoil_line_list = generate_airfoil(airfoil, 0, 0, nr_of_points_bdly_x, lc, with_fan)

    #  Generate the scaled airfoil
    if with_fan:
        scaled_airfoil_line_list = generate_fan_scaled_airfoil(airfoil_scaled, cntr_shift, 0,
                                                               nr_of_points_bdly_x, nr_of_points_bdly_y,
                                                               lc, difference_length, boundary_layer_scale,
                                                               fan_length)
    else:
        scaled_airfoil_line_list = generate_nofan_scaled_airfoil(airfoil_scaled, cntr_shift, 0,
                                                                 nr_of_points_bdly_x, lc)

    # Generate line connecting airfoil and scaled airfoil for transfinite
    transfinite_line_list = generate_line_connecting_airfoil_and_scaled_airfoil(BL_tag, cntr_shift,
                                                                                0, nr_of_points_bdly_y,
                                                                                growth_rate, with_fan,
                                                                                len(airfoil_scaled.x))

    # Different curveloop style for without fan
    if with_fan:
        airfoil_list = airfoil_line_list
    else:
        airfoil_list = curveloop_no_fan(airfoil_line_list, transfinite_line_list, scaled_airfoil_line_list, 0)

    # Generate surface for boundary layer mesh
    BL_surface_cntr = generate_transfinite_mesh(BL_curve_cntr, BL_surf_cntr, cntr_shift, 0, len(airfoil_scaled.x),
                                                len(airfoil.x), with_fan, BL_tag)

    if with_fan:
        gmsh.model.geo.addPlaneSurface([1, 2, cntr_shift + 1], plane_cntr + 3)
    else:
        gmsh.model.geo.addPlaneSurface([2, 1], plane_cntr + 3)

    surface_cntr = np.append(BL_surface_cntr, plane_cntr + 3)

    # Add row of points along the edge of fan to ensure good mesh connectivity between fan and farfield meshes
    if with_fan:
        addpoint_behind_fan(airfoil_scaled, point_tag, 0, lc, plane_cntr)

    if solver == 'su2':
        add_physical_surface(airfoil_list, circle_line_list, surface_cntr)
    elif solver == 'openfoam':
        n1 = len(airfoil.x)
        n_scaled1 = len(airfoil_scaled.x)
        n2 = 0
        n_scaled2 = 0
        n3 = 0
        n_scaled3 = 0
        airfoil_shift = 0
        gmsh_add_physical_surface_openfoam(plane_cntr, with_fan, BL_surf_cntr, n1, n2, n3, n_scaled1,
                                           n_scaled2, n_scaled3, airfoil_shift)

    mesh_success = gmsh_wrapper()
    results = None
    if solver == 'su2':
        results = su2_wrapper(solver_settings.SU2_PATH, AoA, Re/chord_length, chord_length, mu, num_core, rho, mesh_success)
    elif solver == 'openfoam':
        results = openfoam_wrapper(solver_settings.OPENFOAM_PATH, AoA, Re/chord_length, chord_length, mu, num_core, rho, mesh_success)
    gmsh.finalize()

    return results


def gmsh_wrapper(plot_mesh=False):

    mesh_success = False
    try:
        gmsh.model.mesh.generate(3)
        gmsh.write("airfoil_mesh.su2")
        if plot_mesh:
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()
        mesh_success = True
    except FileNotFoundError as err:
        print('Error:', err)
    except Exception as err:
        print("Mesh failed. Error:", err)

    return mesh_success
