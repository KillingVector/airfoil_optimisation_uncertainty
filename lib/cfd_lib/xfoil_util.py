import numpy as np
import os
import subprocess
# import xfoil
# from xfoil.model import Airfoil
from lib.result import Result
import lib.solver_settings as solver_settings


class Xfoil(object):

    def __init__(self, n_crit, transition_location, analysis_type = 1):
        """
        initialisation of the xfoil object
        :param n_crit: critical value for the e^n transition method of xfoil
        """
        self.n_pts = 201  # number of points on the airfoil
        self.n_iter = 250  # number of iterations in the boundary layer solver
        self.n_crit = n_crit
        self.transition_location = transition_location # transition location for boundary layer solver
        self._use_gdes = False  # use xfoil's gdes routine to re-panel the airfoil (should not be needed)
        self.analysis_type = analysis_type

    def write_input(self, airfoil_name, airfoil_state, xfoil_format='set_alpha', identifier=0):
        """
        write the input file to run xfoil automatically. See xfoil documentation for their meaning
        :param airfoil_name: name of the airfoil to be loaded by xfoil
        :param airfoil_state: operating conditions for the airfoil (mach, reynolds, ...)
        :param xfoil_format: set_alpha or set_cl - run xfoil at a given angle or given lift coefficient
        :param identifier: identifier to prevent contamination when running in parallel
        :return: None
        """

        # Open file
        with open('input_' + airfoil_name + '_' + str(identifier) + '.inp', 'w') as f:
            # Load airfoil
            if 'naca' in airfoil_name:
                f.write('{}\n'.format(airfoil_name))
            else:
                f.write('{} {}\n'.format('load', 'airfoils/' + airfoil_name + '_' + str(identifier) + '.txt'))

            # Using the geometry design routine to add points at corners exceeding angle threshold
            if self._use_gdes:
                f.write('{}\n'.format('gdes'))
                f.write('{}\n\n\n\n\n\n'.format('cadd'))

            # # Setting the number of points in the airfoil
            # f.write('{}\n'.format('ppar'))
            # f.write('{} {:d}\n\n\n'.format('n', self.n_pts))
            #
            # # Setting current airfoil panel nodes
            # f.write('{}\n'.format('panel'))

            # Select operating point
            f.write('{}\n'.format('oper'))

            # Setting the boundary layer ncrit value and transition location
            f.write('{}\n'.format('vpar'))
            f.write('{}\n'.format('n'))
            f.write('{:.0f}\n'.format(self.n_crit))
            f.write('{} {:.3f} {:.3f}\n\n'.format('xtr', self.transition_location[0], self.transition_location[1]))

            # Setting the maximum number of iterations
            f.write('{} {:d}\n'.format('iter', self.n_iter))

            # Setting the Mach number
            f.write('{} {:1.2f}\n'.format('mach', airfoil_state.mach))

            # Setting the angle of attack to zero initially
            f.write('{} {:.2f}\n'.format('alfa', 0.0))

            # Setting the Reynolds number
            f.write('{} {:.0f}\n'.format('visc', airfoil_state.reynolds))

            # Setting the analysis
            # Type      Cons Param                  Var     Fixed
            #  1        M,          Re              lift    chord, vel
            #  2        M sqrt(CL), Re sqrt(CL)     vel     chord, lift
            #  3        M,          Re CL           chord   lift, vel !!! not setup
            f.write('{} {:d}\n'.format('type',self.analysis_type))

            # Initialising the boundary layer
            f.write('{}\n{}\n'.format('init', 'init'))
            if xfoil_format == 'set_alpha':
                f.write('{} {:.3f}\n'.format('alfa', airfoil_state.alpha * 180.0 / np.pi))
            elif xfoil_format == 'set_cl':
                f.write('{} {:.3f}\n'.format('cl', airfoil_state.c_l))
            else:
                raise Exception('xfoil format undefined')


            # Enabling polar accumulation
            f.write('{}\n'.format('pacc'))

            # Setting the file name
            f.write('{}\n\n'.format(airfoil_name + '_' + str(identifier) + '.pol'))

            #
            if self.analysis_type == 1:
                # Setting target angle-of-attack or lift coefficient
                if xfoil_format == 'set_alpha':
                    f.write('{} {:.3f}\n'.format('alfa', airfoil_state.alpha * 180.0 / np.pi))
                elif xfoil_format == 'set_cl':
                    f.write('{} {:.3f}\n'.format('cl', airfoil_state.c_l))
                else:
                    raise Exception('xfoil format undefined')
            elif self.analysis_type == 2:
                if airfoil_state.c_l < 0:
                    raise Exception ('xfoil type 2 analysis does not handle c_l < 0')
                else:
                    f.write('{} {:.3f}\n'.format('cl', airfoil_state.c_l))
            else:
                raise Exception('xfoil analysis type undefined or not accepted, must be 1 or 2')

            # Disabling polar accumulation
            f.write('{}\n'.format('pacc'))

            # Quit xfoil
            f.write('\n{}\n'.format('quit'))

            # Close file
            f.close()


    def rotate_airfoil(self, airfoil_name):
        head_tail = os.path.split(airfoil_name)
        foil_name = head_tail[1]

        # Open file
        with open('input_' + foil_name[:-4] + '.rot', 'w') as f:
            # Load airfoil
            # if 'naca' in airfoil_name:
            #     f.write('{}\n'.format(foil_name))
            # else:
            f.write('{} {}\n'.format('load', airfoil_name))

            # go to gdes menu and de-rotate the airfoil
            f.write('{}\n'.format('gdes'))
            f.write('{}\n'.format('dero'))

            # pass de-rotated airfoil to buffer
            f.write('{}\n\n'.format('x'))

            # save de-rotated airfoil
            f.write('{}\n'.format('save'))
            f.write('{}\n'.format(airfoil_name))
            f.write('{}\n'.format('y'))

            # Quit xfoil
            f.write('\n{}\n'.format('quit'))

            # Close file
            f.close()

        # run the process
        subprocess.run([solver_settings.XFOIL_PATH + 'xfoil' + ' < ' + 'input_' + foil_name[:-4] + '.rot'
                        + ' > ' + airfoil_name + '_log.txt'], shell=True)
        # clean up the file
        subprocess.run(['rm -rf ' + 'input_' + foil_name[:-4] + '.rot'],
                       shell=True)
        subprocess.run(['rm -rf ' + airfoil_name + '_log' + '.txt'],
                       shell=True)

    def run(self, airfoil_name, identifier=0):
        """
        run xfoil executable
        :param airfoil_name: name of the file where the coordinates are printed
        :param identifier: identifier of the individual that is being run (to avoid cross-contamination in parallel)
        :return:
        """
        input_file = 'input_' + airfoil_name + '_' + str(identifier) + '.inp'
        subprocess.run([solver_settings.XFOIL_PATH + 'xfoil' + ' < ' + input_file + ' > ' + airfoil_name + '_'
                        + str(identifier) + '_log.txt'], shell=True, timeout=10)

    def read_output(self, airfoil_name, identifier=0):
        """
        read the polar file from an xfoil run
        :param airfoil_name: name of the airfoil
        :param identifier: identifier of the individual that is being run (to avoid cross-contamination in parallel)
        :return:
        """

        # Reading data
        rel_path = airfoil_name + '_' + str(identifier) + '.pol'
        data = []
        if os.path.isfile(rel_path):
            data = np.loadtxt(rel_path, skiprows=12)

        if len(data) > 0:
            alpha = data[0]
            c_l = data[1]
            c_d = data[2]
            c_m = data[4]
        else:
            alpha = 0.0
            c_l = -1.0
            c_d = 1.0
            c_m = 1.0

        # Output
        result = Result(1)
        result.alpha = alpha
        result.c_l = c_l
        result.c_d = c_d
        result.c_m = c_m

        return result

    def cleanup(self, airfoil_name, identifier=0):
        """
        remove all the files that were created during an xfoil run
        :param airfoil_name: name of the airfoil
        :param identifier: identifier of the individual that is being run (to avoid cross-contamination in parallel)
        :return:
        """
        input_file = 'input_' + airfoil_name + '_' + str(identifier) + '.inp'
        subprocess.run(['rm -rf ' + input_file + ' > ' + airfoil_name + '_' + str(identifier) + '_log.txt'],
                       shell=True)
        subprocess.run(['rm -rf ' + airfoil_name + '_' + str(identifier) + '_log' + '.txt'],
                       shell=True)
        subprocess.run(['rm -rf *.bl'], shell=True)


def python_xfoil_wrapper(design, state, xfoil_format, xfoil_manager):
    """
    wrapper to run xfoil through the python API (https://github.com/daniel-de-vries/xfoil-python)
    :param design: design instance
    :param state: flight condition to be analysed
    :param xfoil_format: set_alpha or set_cl - run xfoil at a given angle of attack or at a given lift coefficient
    :param xfoil_manager: contains settings on how to run xfoil (n_iter, n_crit, ...)
    :return:
    """

    # TODO put in a try except call here to avoid the crash

    # Instantiate xfoil instance
    xf = xfoil.XFoil()

    # Silence output
    xf.print = False

    # Set maximum iterations and n_crit
    xf.max_iter = xfoil_manager.n_iter
    xf.n_crit = xfoil_manager.n_crit
    xf.xtr = design.transition_location

    # Generate & assign airfoil coordinates
    airfoil = design.airfoil
    airfoil.generate_section(design.shape_variables, design.n_pts, delta_z_te=0.00025)
    xf.airfoil = xfoil.model.Airfoil(airfoil.x, airfoil.z)

    # Assign Reynolds & Mach numbers
    xf.Re = state.reynolds
    xf.M = state.mach

    # Call python xfoil
    if xfoil_format == 'set_alpha':
        # TODO check if this is in degrees or radians!! - I think it is in radians and so should be converted
        #  as xfoil expects degrees. But make sure this version of xfoil does
        alpha = state.alpha
        c_l, c_d, c_m, _ = xf.a(alpha)
    elif xfoil_format == 'set_cl':
        c_l = state.c_l
        alpha, c_d, c_m, _ = xf.cl(c_l)
    else:
        raise Exception('xfoil format undefined')

    # Output
    if not np.isnan(np.asarray([alpha, c_l, c_d, c_m])).any():
        result = Result(1)
        result.alpha = alpha
        result.c_l = c_l
        result.c_d = c_d
        result.c_m = c_m
    else:
        result = None

    return result


def xfoil_wrapper(design, state, solver, n_core, use_python_xfoil=False, identifier=0, *args, **kwargs):
    """
    wrapper to run xfoil
    :param design: design instance
    :param state: flight condition to be analysed
    :param use_python_xfoil:
    :param identifier:
    :param args: additional arguments
    :param kwargs: keyword arguments. Expects 'use_python_xfoil' to be set to indicate which xfoil path to tak
    :return:
    """
    # use_python_xfoil = kwargs['use_python_xfoil']
    # kwargs.pop('use_python_xfoil', None)
    # # use_python_xfoil = False
    # identifier = kwargs['identifier']

    airfoil_name = design.application_id
    if design.lift_coefficient is not None:
        xfoil_format = 'set_cl'
    elif design.alpha is not None:
        xfoil_format = 'set_alpha'
    else:
        raise Exception('xfoil format undefined')

    # Instantiate solver manager instance
    # TODO set up xtr (set up in design.py and should be passed into here
    xfoil_manager = Xfoil(n_crit=design.n_crit,transition_location=design.transition_location, analysis_type=design.xfoil_analysis_type)

    if use_python_xfoil:
        results = python_xfoil_wrapper(design, state, xfoil_format, xfoil_manager)
    else:
        try:
            # Write input
            xfoil_manager.write_input(airfoil_name, state, xfoil_format=xfoil_format, identifier=identifier)
            # Run solver
            xfoil_manager.run(airfoil_name, identifier=identifier)
            # Read output
            results = xfoil_manager.read_output(airfoil_name, identifier=identifier)
        except:
            pass
        finally:
            # Clean up files
            xfoil_manager.cleanup(airfoil_name, **kwargs)
            # Delete polar file
            subprocess.run(['rm -rf ' + airfoil_name + '_' + str(identifier) + '.pol'],
                           shell=True)
    return results


def xfoil_wrapper_max_lift(design, state, solver, n_core, use_python_xfoil=False, identifier=0, *args, **kwargs):
    """
    wrapper to run xfoil
    :param design: design instance
    :param state: flight condition to be analysed
    :param args: additional arguments
    :param kwargs: keyword arguments. Expects 'use_python_xfoil' to be set to indicate which xfoil path to tak
    :return:
    """

    airfoil_name = design.application_id
    xfoil_format = 'set_alpha'

    # Instantiate solver manager instance
    # TODO set up xtr (set up in design.py and should be passed into here
    xfoil_manager = Xfoil(n_crit=design.n_crit,transition_location=design.transition_location, analysis_type=design.xfoil_analysis_type)

    if use_python_xfoil:
        results = python_xfoil_wrapper(design, state, xfoil_format, xfoil_manager)
    else:
        try:
            # Write input
            xfoil_manager.write_input(airfoil_name, state, xfoil_format=xfoil_format, identifier=identifier)
            # Run solver
            xfoil_manager.run(airfoil_name, identifier=identifier)
            # Read output
            results = xfoil_manager.read_output(airfoil_name, identifier=identifier)
        except:
            pass
        finally:
            # Clean up files
            xfoil_manager.cleanup(airfoil_name, **kwargs)
            # Delete polar file
            subprocess.run(['rm -rf ' + airfoil_name + '_' + str(identifier) + '.pol'],
                           shell=True)
    return results