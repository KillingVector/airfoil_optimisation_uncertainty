# imports
import numpy as np
import os
import math
import lib.solver_settings as solver_settings

# import openfoam write functions
from lib.cfd_lib.openfoam_input_files import GridOptions, write_turbulence_properties, write_transport_properties
from lib.cfd_lib.openfoam_input_files import write_p, write_U, write_omega, write_kl, write_kt, write_nut, write_k
from lib.cfd_lib.openfoam_input_files import write_nuTilda, write_gammaInt, write_ReThetat
from lib.cfd_lib.openfoam_input_files import write_controlDict, write_createPatch, write_createPatch_construct2d
from lib.cfd_lib.openfoam_input_files import write_unsteady_controlDict, write_unsteady_forceCoeffs, \
    write_unsteady_fvSchemes, write_unsteady_fvSolution
from lib.cfd_lib.openfoam_input_files import write_decomposeParDict, write_fieldAverage, write_fvSchemes, \
    write_fvSolution, write_forceCoeffs

# TODO (Shahfiq) clean up naming and add docstring
# TODO (Shahfiq) rename to follow PEP8 convention


class CFDSettings(object):
    def __init__(self):
        self.Solver = 0.0
        self.Iterations = 1000
        self.writeIntervals = self.Iterations
        self.residualControl = '1e-5'
        # Relaxation factor for Pressure
        self.relaxationFactorsP = '0.3'
        # Relaxation factors
        self.relaxationFactors = '0.7'
        self.ToleranceP = '1e-10'
        self.Tolerance = '1e-8'
        self.relTolP = '0.1'
        self.relTol = '0.1'
        self.nSweeps = '1'
        self.nNonOrthogonalCorrectors = '1'
        # Number of times is the pressure term being calculates
        self.nCorrectors = '0'
        self.pRefCell = '0'
        self.pRefValue = '0'
        # Parallel running
        self.NumCores = 0.0
        # Steady (False) or Unsteady (True)
        self.Unsteady = 0.0
        # Settings for unsteady run
        self.UnsteadyDeltaT = '1e-4'
        self.UnsteadyTime = 1.0
        self.UnsteadyWriteInterval = str(self.UnsteadyTime)
        self.ForceWriteIntervalUnsteady = '1e-4'
        self.residualControlUnsteady = '1e-12'
        self.nOuterCorrectors = '100'


class Mesh_settings(object):
    def __init__(self, conditions):
        # y-plus
        self.YPlus = conditions.yPlus
        # Chord Fraction for y-plus ref length
        self.ChordFrac = 1
        # Mesher Type 0-Grid(OGRD) or C-Grid (CGRD)
        self.Grid = 'OGRD'
        # Domain radius (in chord lengths)
        self.Radius = 20
        # Number of points on surface of airfoil
        self.PtsAirfoil = 250
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
            # Ratio of initial spacing wake section of the far field to behind airfoil C-grid (>1 means more initial spacing behind airfoil)
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
    def __init__(self):
        self.rho = 0
        self.chord = 0
        self.mu = 0
        self.Re = 0
        self.aoa = 0
        self.vel = 0
        self.TurbulenceIntensity = 0
        self.airfoil = 0
        self.yPlus = 0


def run(rho_inf, mu, chord_length, Re, AoA, num_cores):

    openfoamLocation = solver_settings.OPENFOAM_PATH

    mesh_name = 'airfoil_mesh.msh'

    # GENERAL SETTINGS
    conditions = Conditions()
    conditions.rho = rho_inf
    conditions.chord = chord_length  # this is the total length of all elements (which will be used to calculate cl, cd, cm)
    conditions.mu = mu
    conditions.Re = Re
    conditions.vel = conditions.Re * conditions.mu / conditions.chord / conditions.rho
    conditions.TurbulenceIntensity = 0.1  # 0.1 = 0.1%
    conditions.airfoil = 'e387'  # dummy variable for GMSH mesh
    conditions.aoa = AoA  # in degrees
    conditions.yPlus = 1  # set this to 30 for wall function capability

    # CFD Settings
    settings = CFDSettings()
    settings.NumCores = num_cores
    settings.Unsteady = False
    settings.Solver = 'kOmegaSSTLM'

    # Builder folder such as 0/constant/system
    FolderBuilder()
    # Build the initial condition files ie. 0/p and the files inside system/ (controlDict,fvSolution,fvSchemes)
    FileBuilder(conditions, settings)
    # convert mesh from GMSH to PolyMesh format
    GMSHToPolyMesh(mesh_name, openfoamLocation)
    # run OpenFOAM
    RunOpenFOAM(settings, openfoamLocation)

    # read files and extract lift, drag and iterations from steady calculations
    finalSteadyIteration, Cl_steady, Cd_steady, Cm_steady = SteadyreadPostProcessing(settings, tolerance=5)
    if settings.Unsteady and finalSteadyIteration == settings.Iterations:
        UnsteadyFolderBuilder(finalSteadyIteration)
        UnsteadyFileBuilder(conditions, settings, finalSteadyIteration)
        UnsteadyRunOpenFOAM(settings, openfoamLocation)
        finalUnsteadyTime, Cl_unsteady, Cd_unsteady, Cm_unsteady = UnsteadyreadPostProcessing(finalSteadyIteration)
        UnsteadyClearFiles(finalSteadyIteration, finalUnsteadyTime, conditions)
        return Cl_unsteady, Cd_unsteady, Cm_unsteady
    else:
        # save files into a result folder
        SteadyClearFiles(finalSteadyIteration)
        return Cl_steady, Cd_steady, Cm_steady


def RunOpenFOAM(CFDSettings, openfoamlocation):
    if CFDSettings.NumCores == 1:
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; simpleFoam > NUL_run')
    else:
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; decomposePar > NUL')
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; mpirun.openmpi -np '
                  + str(CFDSettings.NumCores) + ' simpleFoam -parallel > NUL_run')
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; reconstructPar > NUL')
        os.system('rm -r processor*')


def UnsteadyRunOpenFOAM(CFDSettings, openfoamlocation):
    if CFDSettings.NumCores == 1:
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; pimpleFoam > NUL_run')
    else:
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; decomposePar > NUL')
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; mpirun.openmpi -np '
                  + str(CFDSettings.NumCores) + ' pimpleFoam -parallel > NUL_run')
        os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; reconstructPar > NUL')
        os.system('rm -r processor*')


def SteadyClearFiles(iter):
    commandSystem = 'rm -r geometry.foam 0 constant postProcessing system NUL NUL_run __pycache__ ' + str(iter)
    os.system(commandSystem)


def SteadyreadPostProcessing(CFDSettings, tolerance):
    data = np.loadtxt('postProcessing/forceCoeffs/0/coefficient.dat')
    data_iter = data[:, 0]
    data_cd = data[:, 1]
    data_cl = data[:, 3]
    data_cm = data[:, 5]
    iteration = int(data_iter[-1])
    upper = 1 + tolerance/100
    lower = 1 - tolerance/100
    average_cd = np.mean(data_cd[math.floor(len(data_cd) / 2):-1])
    average_cl = np.mean(data_cl[math.floor(len(data_cl) / 2):-1])
    average_cm = np.mean(data_cm[math.floor(len(data_cm) / 2):-1])
    max_cd = max(data_cd[math.floor(len(data_cd) / 2):-1])
    difference = abs(max_cd/average_cd)

    if iteration == CFDSettings.Iterations:
        if lower <= difference <= upper:
            cl = average_cl
            cd = average_cd
            cm = average_cm
        else:
            cl = None
            cd = None
            cm = None
    elif iteration <= 500:
        cl = None
        cd = None
        cm = None
    else:
        cl = data_cl[-1]
        cd = data_cd[-1]
        cm = data_cm[-1]
    return iteration, cl, cd, cm


def UnsteadyreadPostProcessing(iter):
    path = 'postProcessing/forceCoeffs/' + str(iter) + '/coefficient.dat'
    data = np.loadtxt(path)
    data_iter = data[:, 0]
    data_cd = data[:, 1]
    data_cl = data[:, 3]
    data_cm = data[:, 5]
    iter = float(data_iter[-1])
    cl = np.mean(data_cl[math.floor(len(data_cl) / 2):-1 - 20])
    cd = np.mean(data_cd[math.floor(len(data_cd) / 2):-1 - 20])
    cm = np.mean(data_cm[math.floor(len(data_cm) / 2):-1 - 20])
    return iter, cl, cd, cm


def UnsteadyClearFiles(iter_steady, iter_unsteady, conditions):
    os.system('mkdir pimpleFoam')
    commandSystem = 'mv 0 constant system geometry.foam yPlus NUL NUL_run postProcessing ' + str(iter_steady) + ' ' + str(math.floor(iter_unsteady)) + '* pimpleFoam'
    os.system(commandSystem)
    commandSystem = 'mkdir results_' + str(conditions.aoa)
    os.system(commandSystem)
    os.system('rm -r pimpleFoam simpleFoam results_' + str(conditions.aoa))


def MesherSettings(conditions):

    settings = Mesh_settings(conditions)
    GridOptions(conditions, settings)
    return settings


def construct2dMeshBuilder(conditions, construct2dLocation):
    cwd = os.getcwd()
    commandSystem = 'cp -r airfoils/' + conditions.airfoil + '.dat ' + cwd
    os.system(commandSystem)
    commandSystem = 'ln -s ' + construct2dLocation
    os.system(commandSystem)
    commandSystem = './construct2d ' + conditions.airfoil + '.dat < construct2dCommands.txt > NUL'
    os.system(commandSystem)
    commandSystem = 'rm -r grid_options.in construct2dCommands.txt construct2d ' + conditions.airfoil + '.dat ' + conditions.airfoil + '.nmf ' + conditions.airfoil + '_stats.p3d'
    os.system(commandSystem)


def construct2dToPolyMesh(Mesher, mesh_name, openfoamlocation):
    write_createPatch_construct2d()
    commandSystem = 'export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; plot3dToFoam ' \
                    + mesh_name + ' -' + str(Mesher.GDIM) + 'D 1 -singleBlock -noBlank > NUL'
    os.system(commandSystem)
    os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; autoPatch 45 -overwrite > NUL')
    os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; createPatch -overwrite > NUL')
    commandSystem = 'rm -r ' + mesh_name
    os.system(commandSystem)
    os.system('touch geometry.foam')


def FolderBuilder():
    os.system('mkdir system')
    os.system('mkdir 0')
    os.system('mkdir constant')


def GMSHToPolyMesh(mesh_name, openfoamlocation):
    # convert mesh from gmsh to polymesh
    os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; gmshToFoam ' + mesh_name + ' > NUL')
    # write createPatch file
    write_createPatch()
    os.system('export LD_LIBRARY_PATH=""; . ' + openfoamlocation + '; createPatch -overwrite > NUL')
    os.system('touch geometry.foam')


def FileBuilder(conditions, CFDSettings):
    # write files in constant folder
    write_turbulence_properties(CFDSettings)
    write_transport_properties(conditions)
    # write files in 0 folder
    write_p()
    write_U(conditions)
    write_nut(CFDSettings, conditions)
    if CFDSettings.Solver == 'SpalartAllmaras':
        write_nuTilda()
    elif CFDSettings.Solver == 'kOmegaSST' or CFDSettings.Solver == 'kOmegaSSTLowRe':
        write_k(conditions)
        write_omega(conditions, CFDSettings)
    elif CFDSettings.Solver == 'kOmegaSSTLM':
        write_k(conditions)
        write_omega(conditions, CFDSettings)
        write_gammaInt()
        write_ReThetat(conditions)
    elif CFDSettings.Solver == 'kkLOmega':
        write_kt(conditions)
        write_kl()
        write_omega(conditions, CFDSettings)
    # write files in system folder
    write_controlDict(CFDSettings)
    write_decomposeParDict(CFDSettings)
    write_fvSchemes(CFDSettings)
    write_fvSolution(CFDSettings)
    write_forceCoeffs(conditions)


def UnsteadyFolderBuilder(iter):
    os.system('mkdir simpleFoam')
    commandSystem = 'cp -r 0 constant system geometry.foam postProcessing NUL NUL_run ' + str(iter) + ' simpleFoam'
    os.system(commandSystem)


def UnsteadyFileBuilder(conditions, CFDSettings, final_iter):
    write_unsteady_controlDict(CFDSettings, final_iter)
    write_unsteady_fvSchemes(CFDSettings)
    write_unsteady_fvSolution(CFDSettings)
    write_fieldAverage(CFDSettings)
    write_unsteady_forceCoeffs(conditions, CFDSettings)

