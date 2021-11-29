import csv
import os
import numpy as np
from lib.result import SU2Results as Results
from lib.cfd_lib.su2_input_files import write_cfg_steady_compressible_file, write_cfg_steady_incompressible_file

# TODO (Shahfiq) clean up naming and add docstring
# TODO (Shahfiq) rename to follow PEP8 convention


class Conditions(object):
    def __init__(self):
        self.compressible = False
        self.chord = 1
        self.Re = 0
        self.aoa = 0
        self.vel = 0
        self.TurbulenceIntensity = 0
        self.airfoil = ''
        # fluid type
        self.mach = 0
        self.mu = 1.789e-5
        self.rho = 1.225
        self.gamma = 1.4
        self.R = 287.05  # air
        self.T = 288.15  # sea level in kelvin
        self.pressure = 101325  # approx sea level Pa
        self.a = np.sqrt(self.gamma * self.R * self.T)  # speed of sound
# TODO link this to the atmospheric model in flight.py


class CFDSettings(object):
    def __init__(self):
        self.Solver = 0.0
        self.Trans = 0.0
        self.Iterations = 0.0
        self.residualControl = 0.0
        # Parallel running
        self.NumCores = 0.0
        # mesher
        self.mesh = 0.0


def run(path, rho, mu, chord_length, Re, AoA, Mach, num_core, mesher):
    """
    function to run SU2 CFD calculations
    :param path: path to the SU2 solver. This is set in solver_settings.py. Do not change it here!
    :param flight_condition: conditions to run the calculation
    :param chord_length: length of the airfoil
    :param mesher: selected mesher (construct2D or gmsh)
    :param num_core: number of cores used in a single CFD call
    :return:
    """
    # Set the operating conditions and some solver settings
    # TODO add a flag in design.py to choose compressible or incompressible
    conditions = Conditions()
    conditions.compressible = True
    conditions.Re = Re
    conditions.TurbulenceIntensity = 0.1  # 0.1 = 0.1%
    conditions.airfoil = 'airfoil_mesh'
    conditions.aoa = AoA/np.pi*180
    conditions.chord = chord_length
    conditions.mu = mu
    conditions.rho = rho
    if conditions.compressible:
        conditions.mach = Mach

    # CFD settings such as solver/tolerance etc initialisation
    # TODO add a flag in design.py to choose turbulence model and transition
    CFD_settings = CFDSettings()
    CFD_settings.Solver = 'SST'
    CFD_settings.Trans = 'NONE'
    CFD_settings.residualControl = -8
    CFD_settings.Iterations = 2500
    CFD_settings.NumCores = num_core
    CFD_settings.multigrid = 3  # (0,1,2,3 options)
    CFD_settings.mesh = mesher

    if not conditions.compressible:
        conditions.vel = conditions.Re * conditions.mu / conditions.chord / conditions.rho
    else:
        conditions.vel = conditions.a * conditions.mach
        conditions.mu = conditions.vel * conditions.rho * conditions.chord / conditions.Re

    if mesher == 'construct2d':
        mesh_name = conditions.airfoil + '.p3d'
        Plot3dToSU2(mesh_name)

    # run the actual call to SU2, read the results and clear up the files
    runSU2(conditions, CFD_settings, path)
    cl, cd, cm = SteadyreadPostProcessing(CFD_settings, conditions)
    ClearFiles()
    return cl, cd, cm


def runSU2(conditions, CFD_settings, path):
    """
    the actual call to run the SU2 solver
    :param conditions: instance of the Conditions class
    :param CFD_settings: instance of the CFD_settings class
    :param path: path to the SU2 solver
    :return:
    """
    if not conditions.compressible:
        write_cfg_steady_incompressible_file(conditions, CFD_settings)
    else:
        write_cfg_steady_compressible_file(conditions, CFD_settings)

    if CFD_settings.NumCores == 1:
        os.system(path + ' config_CFD.cfg > log')
    else:
        systemCommand = 'mpirun.mpich -np ' + str(CFD_settings.NumCores) + ' ' + path + ' config_CFD.cfg > log'
        os.system(systemCommand)

    os.system('rm -r config_CFD.cfg')


def process_call(results, max_iter, residual_tolerance):
    """
    process the results
    :param results: instance of the Results class
    :param max_iter: maximum iterations allowed for the solver
    :param residual_tolerance: tolerance for the residuals (to determine if solution is converged or not)
    :return: cl, cd, cm: lift, drag and pitching moment coefficients
    """
    if results.residual > 0:
        print('residual blew up')
        cl = None
        cd = None
        cm = None
    elif abs(residual_tolerance) > abs(results.residual):
        print('did not reach tolerance')
        cl = None
        cd = None
        cm = None
    elif results.iter >= (max_iter - 2):
        print('max iterations exceeded')
        cl = None
        cd = None
        cm = None
    else:
        cl = results.cl
        cd = results.cd
        cm = results.cm
    return cl, cd, cm


def SteadyreadPostProcessing(CFD_settings, conditions):
    result = Results()
    if os.path.isfile('history.csv'):
        with open('history.csv', 'r') as csv_file:
            for line in list(csv.reader(csv_file)): pass
            if conditions.compressible:
                if CFD_settings.Solver == 'SST':
                    result.iter = float(line[2])
                    result.residual = float(line[3])
                    result.cd = float(line[9])
                    result.cl = float(line[10])
                    result.cm = float(line[14])
                else:
                    result.iter = float(line[2])
                    result.residual = float(line[3])
                    result.cd = float(line[8])
                    result.cl = float(line[9])
                    result.cm = float(line[13])
            elif not conditions.compressible:
                if CFD_settings.Solver == 'SST':
                    result.iter = float(line[2])
                    result.residual = float(line[3])
                    result.cd = float(line[8])
                    result.cl = float(line[9])
                    result.cm = float(line[13])
                else:
                    result.iter = float(line[2])
                    result.residual = float(line[3])
                    result.cd = float(line[7])
                    result.cl = float(line[8])
                    result.cm = float(line[12])

        cl, cd, cm = process_call(result, CFD_settings.Iterations, CFD_settings.residualControl)

    else:
        print('History file does not exist - SU2 crashed?')
        print('cwd:', os.getcwd())
        cl = None
        cd = None
        cm = None

    return cl, cd, cm


def ClearFiles():
    os.system('rm -r surface_flow.vtu flow.vtu restart_flow.dat history.csv log *.su2')


def Plot3dToSU2(filename_p3d, delete=True):
    p2d_File = open(filename_p3d, "r")
    # Read the body
    body = p2d_File.read().replace("\n", " ").replace("\t", " ").split()
    p2d_File.close()

    nNode = int(body[0])
    mNode = int(body[1])
    body.remove(body[0])
    body.remove(body[0])

    # Write the .su2 file
    filename = filename_p3d.rsplit(".", 1)[0] + ".su2"
    su2_File = open(filename, "w")

    # Write the header
    su2_File.write("NDIME=2\n")
    su2_File.write("NELEM=%s\n" % ((mNode - 1) * (nNode - 1)))

    # Write the connectivity
    iElem = 0
    for jNode in range(mNode - 1):
        for iNode in range(nNode - 1):

            Point0 = iNode + (jNode * (nNode - 1))
            Point1 = (iNode + 1) + (jNode * (nNode - 1))
            Point2 = (iNode + 1) + (jNode + 1) * (nNode - 1)
            Point3 = iNode + (jNode + 1) * (nNode - 1)

            if iNode == nNode - 2:
                Point1 = jNode * (nNode - 1)
                Point2 = (jNode + 1) * (nNode - 1)

            su2_File.write("9 \t %s \t %s \t %s \t %s \t %s\n" % (Point0, Point1, Point2, Point3, iElem))
            iElem = iElem + 1

    # Write the coordinates
    nPoint = (nNode) * (mNode)
    su2_File.write("NPOIN=%s\n" % ((nNode - 1) * (mNode)))
    iPoint = 0
    for jNode in range(mNode):
        for iNode in range(nNode):
            XCoord = body[jNode * nNode + iNode]
            YCoord = body[(nNode * mNode) + jNode * nNode + iNode]

            if iNode != (nNode - 1):
                su2_File.write("%s \t %s \t %s\n" % (XCoord, YCoord, iPoint))
                iPoint = iPoint + 1

    # Write the boundaries
    su2_File.write("NMARK=2\n")

    su2_File.write("MARKER_TAG= airfoil\n")
    points = (nNode - 1)
    FirstPoint = 1
    su2_File.write("MARKER_ELEMS=%s\n" % points)
    for iNode in range(points - 1):
        su2_File.write(
            "3 \t %s \t %s\n" % (FirstPoint + iNode - 1, FirstPoint + iNode + 1 - 1))  # minus 1 because C++ uses 0
    su2_File.write("3 \t %s \t %s\n" % (FirstPoint + points - 1 - 1, FirstPoint - 1))  # minus 1 because C++ uses 0

    su2_File.write("MARKER_TAG= farfield\n")
    points = (nNode - 1)
    elems = points
    FirstPoint = 1 + (nNode - 1) * (mNode - 1)
    su2_File.write("MARKER_ELEMS=%s\n" % points)
    for iNode in range(points - 1):
        su2_File.write(
            "3 \t %s \t %s\n" % (FirstPoint + iNode - 1, FirstPoint + iNode + 1 - 1))  # minus 1 because C++ uses 0
    su2_File.write("3 \t %s \t %s\n" % (FirstPoint + points - 1 - 1, FirstPoint - 1))  # minus 1 because C++ uses 0

    su2_File.close()
    if delete:
        os.system('rm -r ' + filename_p3d)
