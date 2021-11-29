import os
import lib.solver_settings as solver_settings

from lib.cfd_lib.util import su2_wrapper, openfoam_wrapper


class MeshSettings(object):
    """
    contains all the settings for a construct2D mesh. See construct2D for their meaning
    """
    def __init__(self):
        # y-plus
        self.YPlus = 1
        # Chord Fraction for y-plus ref length
        self.ChordFrac = 1
        # Mesher Type 0-Grid(OGRD) or C-Grid (CGRD)
        self.Grid = 'OGRD'
        # Domain radius (in chord lengths)
        self.Radius = 20
        # Number of points on surface of airfoil
        self.PtsAirfoil = 501
        # Number of points on normal to surface of airfoil
        self.PtsNormal = 100
        # Leading Edge spacing
        self.LESp = 4*10**-4
        # Trailing Edge spacing
        self.TESp = 2.5*10**-4
        # C-Grid Wake Points (only required for C-Grid)
        self.CWakePoints = 50
        # Extrusion method hyperbolic(HYPR) or Elliptical(ELLP)
        self.extrMethod = 'HYPR'
        # Grid dimension 2D or 3D
        self.GDIM = 2
        # Number of planes to be created for 3D output
        self.NPLN = 1
        # Spacing between extruded planes for a 3D output grid. Cannot be 0
        self.DPLN = 1
        # Fun3d Compatible (T/F)
        self.f3dm = 'F'

        # Extra settings if ELLP or HYPR extrusion is chosen
        if self.extrMethod == 'ELLP':
            # ELLP Settings
            # Clustering of grid points on O-grid farfield (1=uniform)
            self.FDST = 1
            # Ratio of length of the wake section of the far field to behind airfoil C-grid (1=uniform)
            self.FWKL = 1
            # Ratio of initial spacing wake section of the far field to behind airfoil C-grid (>1 means more initial
            # spacing behind airfoil)
            self.FWKI = 10
            # Initial smoothing steps for grid generation
            self.STP1 = 1000
            # Final smoothing steps for grid generation
            self.STP2 = 20
            # First point from TE on top Surface (Increase to reduce skewness)
            self.NRMT = 1
            # First point from TE on bottom Surface (Increase to reduce skewness)
            self.NRMB = 1

            # Place Holders When ELLP is chosen
            self.ALFA = 1
            self.EPSI = 15
            self.EPSE = 0
            self.FUNI = 0.2
            self.ASMT = 20
        elif self.extrMethod == 'HYPR':
            # HYPR Settings
            # Implicitness parameter for hyperbolic marching (1 is good, increase to prevent grid lines crossing)
            self.ALFA = 1
            # Implicit smoothing parameter. Smooths grid and adds stability (don't go to high)
            self.EPSI = 15
            # Explicit smoothing parameter. Much more unstable than EPSI. (0 is best)
            self.EPSE = 0
            # Number of cell area smoothing iterations (can help smooth grid)
            self.ASMT = 200
            # Uniformness of cell areas near the farfield (1=completely uniform,0=opposite) (OGRD-0.2,CGRD-0.025)
            if self.Grid == 'OGRD':
                self.FUNI = 0.2
            else:
                self.FUNI = 0.025

            # Place Holders When HYPR is chosen
            self.FDST = 1
            self.FWKL = 1
            self.FWKI = 10
            self.STP1 = 1000
            self.STP2 = 20
            self.NRMT = 1
            self.NRMB = 1


class Conditions(object):
    """
    class that contains all the conditions for the CFD solver.
    """
    def __init__(self):
        self.rho = 0.0
        self.chord = 0.0
        self.mu = 0.0
        self.Re = 0.0
        self.aoa = 0.0
        self.vel = 0.0
        self.TurbulenceIntensity = 0.0
        self.airfoil = ''


def run_single_element(design, flight_condition, solver, num_core, identifier=0, **kwargs):
    """
    sets up a run for a single element airfoil
    :param design: the particular combination of design variables that determine that airfoil shape and Aangle of attack
    :param flight_condition: conditions where the airfoil is evaluated (density / viscosity / Reynolds nr /
            angle of attack
    :param solver: which CFD solver is used (SU2 or openfoam)
    :param num_core: how many cores are used by the CFD solver
    :param identifier: used to differentiate between cores when running in parallel - to avoid one solution being over-
                        written by a different core
    :param kwargs: additional arguments needed
    :return: results: instance of the result class that contains the aerodynamic results of the solver
    """
    # set location of construct 2D - do not change here. needs to be set in solver_settings.py
    construct2d_location = solver_settings.CONSTRUCT2D_PATH

    # atmospheric properties
    rho = flight_condition.ambient.rho
    mu = flight_condition.ambient.viscosity.mu
    Re = flight_condition.reynolds
    AoA = flight_condition.alpha
    Mach = flight_condition.mach
    chord_length = 1

    # set the conditions to be passed to the mesher
    conditions = Conditions()
    conditions.Re = Re
    conditions.airfoil = 'airfoils/' + design.application_id + '_' + str(identifier)

    # set up the mesher and mesh the airfoil
    mesher_settings(conditions)
    mesh_success = construct2d_wrapper(conditions, construct2d_location)
    mesher = 'construct2d'

    # prepare for the CFD solver
    results = None
    if solver.lower() == 'su2':
        results = su2_wrapper(solver_settings.SU2_PATH, AoA, Re, chord_length, mu, num_core, rho, Mach, mesher, mesh_success)
    elif solver.lower() == 'openfoam':
        results = openfoam_wrapper(solver_settings.OPENFOAM_PATH, AoA, Re, chord_length, mu, num_core, rho, mesh_success)
    return results


def construct2d_wrapper(conditions, construct2d_location):
    """
    wrapper to call the construct2D mesher for single element airfoils. Can not be used for multiple elements
    currently set up to produce an O-grid only. A small trailing edge thickness is needed so that an o-grid can be
    produced

    :param conditions: conditions where the airfoil is evaluated (density / viscosity / Reynolds nr /
            angle of attack
    :param construct2d_location: location of the construct2D mesher executable. Needs to be set in solver_settings.py
    :return: mesh_success. Boolean to indicate if the mesher was successful or not. Does not indicate the quality of the
            mesh! only if the solver exited correctly or not
    """
    mesh_success = False
    try:
        commandSystem = 'ln -s ' + construct2d_location
        os.system(commandSystem)
        commandSystem = './construct2d ' + conditions.airfoil + '.txt < construct2dCommands.txt > NUL'
        os.system(commandSystem)
        commandSystem = 'rm -r grid_options.in construct2dCommands.txt construct2d NUL '\
                        + conditions.airfoil + '.nmf ' + conditions.airfoil + '_stats.p3d'
        os.system(commandSystem)
        commandSystem = 'mv ' + conditions.airfoil + '.p3d airfoil_mesh.p3d'
        os.system(commandSystem)
        mesh_success = True
    except FileNotFoundError as err:
        print('Error:', err)
        commandSystem = 'rm -r grid_options.in construct2dCommands.txt construct2d '
        os.system(commandSystem)
    except Exception as err:
        print("Mesh failed. Error:", err)
        commandSystem = 'rm -r grid_options.in construct2dCommands.txt construct2d '
        os.system(commandSystem)
    return mesh_success

def mesher_settings(conditions):
    """
    Create the settings for the mesher (so that angle dependent variables are set correctly)
    :param conditions: conditions where the airfoil is evaluated (density / viscosity / Reynolds nr /
            angle of attack
    :return: settings: an instance of the Mesh_settings class
    """
    settings = MeshSettings()
    grid_options(conditions, settings)
    return settings


def grid_options(conditions, mesh_settings):
    """
    Create the grid_options.in file that contains all the required parameters to successfully create a mesh and
    create the input file for construct2D
    :param conditions: conditions where the airfoil is evaluated (density / viscosity / Reynolds nr /
            angle of attack
    :param mesh_settings: an instance of the Mesh_settings class
    :return: None
    """
    f = open('grid_options.in', 'w')
    f.write('&SOPT \n')
    f.write('nsrf = ' + str(mesh_settings.PtsAirfoil) + ' \n')
    f.write('lesp = ' + str(mesh_settings.LESp) + ' \n')
    f.write('tesp = ' + str(mesh_settings.TESp) + ' \n')
    f.write('radi = ' + str(mesh_settings.Radius) + ' \n')
    f.write('nwke = ' + str(mesh_settings.CWakePoints) + ' \n')
    f.write('fdst = ' + str(mesh_settings.FDST) + ' \n')
    f.write('fwkl = ' + str(mesh_settings.FWKL) + ' \n')
    f.write('fwki = ' + str(mesh_settings.FWKI) + ' \n')
    f.write('/ \n')
    f.write('&VOPT \n')
    f.write('name = "' + conditions.airfoil + '" \n')
    f.write('jmax = ' + str(mesh_settings.PtsNormal) + ' \n')
    f.write('slvr = "' + mesh_settings.extrMethod + '" \n')
    f.write('topo = "' + mesh_settings.Grid + '" \n')
    f.write('ypls = ' + str(mesh_settings.YPlus) + ' \n')
    f.write('recd = ' + str(conditions.Re) + ' \n')
    f.write('cfrc = ' + str(mesh_settings.ChordFrac) + ' \n')
    f.write('stp1 = ' + str(mesh_settings.STP1) + ' \n')
    f.write('stp2 = ' + str(mesh_settings.STP2) + ' \n')
    f.write('nrmt = ' + str(mesh_settings.NRMT) + ' \n')
    f.write('nrmb = ' + str(mesh_settings.NRMB) + ' \n')
    f.write('alfa = ' + str(mesh_settings.ALFA) + ' \n')
    f.write('epsi = ' + str(mesh_settings.EPSI) + ' \n')
    f.write('epse = ' + str(mesh_settings.EPSE) + ' \n')
    f.write('funi = ' + str(mesh_settings.FUNI) + ' \n')
    f.write('asmt = ' + str(mesh_settings.ASMT) + ' \n')
    f.write('/ \n')
    f.write('&OOPT \n')
    f.write('gdim = ' + str(mesh_settings.GDIM) + ' \n')
    f.write('npln = ' + str(mesh_settings.NPLN) + ' \n')
    f.write('dpln = ' + str(mesh_settings.DPLN) + ' \n')
    f.write('f3dm = ' + mesh_settings.f3dm + '\n')
    f.write('/ \n')
    f.close()

    f = open('construct2dCommands.txt', 'w')
    f.write('grid \n')
    f.write('smth \n')
    f.write('quit \n')
    f.close()
