import os

from lib.cfd_lib import su2_util, openfoam_util, xfoil_util


class Result(object):
    def __init__(self):
        self.c_l = 0.0
        self.c_d = 0.0
        self.c_m = 0.0


def su2_wrapper(path, AoA, Re, chord_length, mu, num_core, rho, Mach, mesher, mesh_success=False):

    results = Result()

    if mesh_success:
        try:
            su2_rel_location = 'SU2_CFD'
            cl, cd, cm = su2_util.run(path + su2_rel_location, rho, mu, chord_length, Re, AoA, Mach, num_core, mesher)
            if cl is None:
                results = None
            else:
                results.c_l = cl / chord_length
                results.c_d = cd / chord_length
                results.c_m = cm / chord_length
            if mesher == 'gmsh':
                os.system('rm -r airfoil_mesh.msh')
        except Exception as err:
            print("SU2 failed. Error:", err)
            results = None
    else:
        results = None

    return results


def openfoam_wrapper(path, AoA, Re, chord_length, mu, num_core, rho, mesh_success=False):

    results = Result()

    if mesh_success:
        try:
            openfoam_rel_location = 'openfoam/openfoam2012/etc/bashrc'
            cl, cd, cm = openfoam_util.run(path + openfoam_rel_location, rho, mu, chord_length, Re, AoA, num_core)
            if cl is None:
                results = None
            else:
                results.c_l = cl
                results.c_d = cd
                results.c_m = cm
            os.system('rm -r airfoil_mesh.msh')
        except Exception as err:
            print("OpenFOAM failed. Error:", err)
            results = None
    else:
        results = None

    return results


