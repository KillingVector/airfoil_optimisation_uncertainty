import numpy as np
import os
import subprocess
import lib.solver_settings as solver_settings
from lib.result import Result, Result2
from lib.airfoil import Airfoil


class MSES(object):

    def __init__(self, n_crit, transition_location, approach='set_alpha',):

        self.n_pts = 201

        # Maximum number of iterations
        self.n_iter = 200
        self.n_iter_two_element = 10

        # Left inlet, right outlet, lowest & uppermost gridlines
        self.bounds = [-2.25, 3.25, -2.5, 3.0]

        # Global variables
        # Variable 3: Far-field vortex strength
        # Variable 4: Freestream angle of attack
        # Variable 5: LE stagnation point for all elements & mass
        # fraction for all-1 elements
        # Variable 7: x,y-doublet,source strengths
        # Variable 10: Reynolds number DOF
        self.variables = [3, 4, 5, 7, 10]

        # Setting global constraints
        # Constraint 3: Set LE Kutta condition for all elements
        # Constraint 4: Set TE Kutta condition for all elements
        # Constraint 5: Drive alpha to ALFAIN
        # Constraint 6: Drive c_l to CLIFIN
        # Constraint 7: Drive x,y-doublets,source to farfield match
        # Constraint 17: Drive Reynolds number to REYNIN
        if approach == 'set_alpha':
            self.constraints = [3, 4, 5, 7, 17]
        elif approach == 'set_cl':
            self.constraints = [3, 4, 6, 7, 17]

        # Critical amplification factor n for e**n transition model (n = 9 is the standard model)
        self.n_crit = n_crit

        # transition location
        self.transition_location = transition_location

        # Equation
        # ISMOM = 3: Use s-momentum equation, with isentropic condition only near the leading edge
        self.ismom = 3

        # Boundary conditions
        # IFFBC = 2: Use vortex+source+doublet airfoil farfield boundary conditions (best model for infinite domains)
        self.iffbc = 2

        # Transition locations
        self.xtr_top = self.transition_location[0]
        self.xtr_bottom = self.transition_location[1]

        # Critical Mach number above which artificial dissipation is added
        self.m_crit = 0.98
        # Artificial dissipation constant
        self.m_ucon = 1.0

        # Indicates which side(s) get altered in mixed-inverse cases (ignored in direct cases)
        self.ismove = 1
        # Indicates the pressure boundary condition imposed in mixed-inverse cases (ignored in direct cases)
        self.ispres = 1

        # Modal geometry and element position DOFs (only used if the DPOSn flag (30) is chosen as a DOF
        self.n_modn = 0
        self.n_posn = 0

    def write_blade(self, airfoil_name, airfoil=None, identifier=0):

        if airfoil is None:

            airfoil = Airfoil(self.n_pts)

            # Load airfoil
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rel_path = 'airfoils/' + airfoil_name + '_' + str(identifier) + '.txt'
            abs_path = os.path.join(script_dir[:-4], rel_path)

            data = np.loadtxt(abs_path, skiprows=1)
            airfoil.x, airfoil.z = data[:, 0], data[:, 1]

        with open('blade.' + airfoil_name + '_' + str(identifier), 'w') as f:

            # Airfoil name
            f.write('{}\n'.format(airfoil_name + '_' + str(identifier)))

            # Domain bounds
            f.write('{:2.2f} {:2.2f} {:2.2f} {:2.2f}\n'.format(*self.bounds))

            # Airfoil coordinates
            for i in range(len(airfoil.x)):
                f.write('{:.16f} {:.16f}\n'.format(airfoil.x[i], airfoil.z[i]))
            f.write('{:d} {:d}\n'.format(999, 999))

            f.close()

    def write_blade_two_element(self, airfoil_name, airfoil1=None, airfoil2=None, identifier=0):

        if airfoil1 is None:
            airfoil1 = Airfoil(self.n_pts)

            # Load airfoil
            data = np.loadtxt('./airfoils/' + airfoil_name + '_' + str(identifier) + '.txt', skiprows=1)
            airfoil1.x, airfoil1.z = data[:, 0], data[:, 1]

        if airfoil2 is None:
            airfoil2 = Airfoil(self.n_pts)

            # Load airfoil
            data = np.loadtxt('./airfoils/' + airfoil_name + '_' + str(identifier) + '_element2.txt', skiprows=1)
            airfoil2.x, airfoil2.z = data[:, 0], data[:, 1]

        with open('blade.' + airfoil_name + '_' + str(identifier), 'w') as f:

            # Airfoil name
            f.write('{}\n'.format(airfoil_name + '_' + str(identifier)))

            # Domain bounds
            f.write('{:2.2f} {:2.2f} {:2.2f} {:2.2f}\n'.format(*self.bounds))

            # Airfoil coordinates
            for i in range(len(airfoil1.x)):
                f.write('{:.16f} {:.16f}\n'.format(airfoil1.x[i], airfoil1.z[i]))
            f.write('{:d} {:d}\n'.format(999, 999))
            for i in range(len(airfoil2.x)):
                f.write('{:.16f} {:.16f}\n'.format(airfoil2.x[i], airfoil2.z[i]))
            f.close()

    def write_mset_input(self, airfoil_name, airfoil_state, identifier=0):

        with open('mset_input_' + airfoil_name + '_' + str(identifier) + '.txt', 'w') as f:

            # Specifying angle of attack
            f.write('{:d}\n'.format(1))
            f.write('{:2.2f}\n'.format(airfoil_state.alpha*180.0/np.pi))

            # Initial surface gridding
            f.write('{:d}\n'.format(2))

            # # Airfoil upper surface gridding refinement
            f.write('{}\n'.format('U'))
            f.write('{:2.4f},{:2.4f},{:2.3f}\n\n'.format(0.0100, 0.0500, 0.500))

            # Option 3 invokes grid smoothing, option 4 writes the solution file mdat.xxx and option 8 writes the
            # grid parameter file gridpar.xxx
            f.write('{:d}\n{:d}\n{:d}\n{:d}\n'.format(3, 4, 8, 0))

            f.close()

    def write_mset_input_sweep(self, airfoil_name, airfoil_state, identifier=0):

        with open('mset_input_' + airfoil_name + '_' + str(identifier) + '.txt', 'w') as f:
            # Specifying angle of attack
            f.write('{:d}\n'.format(1))
            f.write('{:2.2f}\n'.format(airfoil_state[0].alpha*180.0/np.pi))

            # Initial surface gridding
            f.write('{:d}\n'.format(2))

            # # Airfoil upper surface gridding refinement
            f.write('{}\n'.format('U'))
            f.write('{:2.4f},{:2.4f},{:2.3f}\n\n'.format(0.0100, 0.0500, 0.500))

            # Option 3 invokes grid smoothing, option 4 writes the solution file mdat.xxx and option 8 writes the
            # grid parameter file gridpar.xxx
            f.write('{:d}\n{:d}\n{:d}\n{:d}\n'.format(3, 4, 8, 0))

            f.close()

    def write_mset_input_two_element(self, airfoil_name, airfoil_state, identifier=0):

        with open('mset_input_' + airfoil_name + '_' + str(identifier) + '.txt', 'w') as f:
            # Specifying angle of attack
            f.write('{:d}\n'.format(1))
            f.write('{:2.2f}\n'.format(airfoil_state.alpha*180.0/np.pi))

            # Initial surface gridding
            f.write('{:d}\n'.format(2))

            # # Airfoil upper surface gridding refinement
            f.write('{}\n'.format('U'))
            f.write('{:2.4f},{:2.4f},{:2.3f}\n\n'.format(0.0100, 0.0500, 0.500))

            # # Airfoil upper surface gridding refinement
            f.write('{}\n'.format('U'))
            f.write('{:2.4f},{:2.4f},{:2.3f}\n\n'.format(0.0100, 0.0500, 0.500))

            # Option 3 invokes grid smoothing, option 4 writes the solution file mdat.xxx and option 8 writes the
            # grid parameter file gridpar.xxx
            f.write('{:d}\n{:d}\n{:d}\n{:d}\n'.format(3, 4, 8, 0))

            f.close()

    def write_mses(self, airfoil_name, airfoil_state, approach='set_alpha', identifier=0):

        with open('mses.' + airfoil_name + '_' + str(identifier), 'w') as f:

            # Variables
            str_fmt = ''.join(['{:d} ' for _ in range(len(self.variables))]) + '\n'
            f.write(str_fmt.format(*self.variables))
            # Constraints
            str_fmt = ''.join(['{:d} ' for _ in range(len(self.constraints))]) + '\n'
            f.write(str_fmt.format(*self.constraints))

            # Mach number, lift coefficient & angle of attack
            if approach == 'set_alpha':
                f.write('{:1.2f} {:2.3f} {:2.3f} \t\t\t |\t{} {} {}\n'.format(airfoil_state.mach, 0.0,
                                                                              airfoil_state.alpha*180.0/np.pi,
                                                                              'MACH', 'CL', 'ALFA'))
            elif approach == 'set_cl':
                f.write('{:1.2f} {:2.3f} {:2.3f} \t\t\t |\t{} {} {}\n'.format(airfoil_state.mach, airfoil_state.c_l,
                                                                              0.0,
                                                                              'MACH', 'CL', 'ALFA'))

            # Governing equation & boundary condition
            f.write('{:d} {:d} \t\t\t\t\t |\t{} {}\n'.format(self.ismom, self.iffbc, 'ISMOM', 'IFFBC'))

            # Reynolds number & n-crit
            f.write('{:.0f} {:.1f} \t\t\t |\t{} {}\n'.format(airfoil_state.reynolds, self.n_crit, 'REYN', 'NCRIT'))

            # Transition locations
            str_fmt = ''.join(['{:1.3f} ' for _ in range(2)]) + '\t\t\t |\t{} {}' + '\n'
            f.write(str_fmt.format(self.xtr_top, self.xtr_bottom, 'XTR1', 'XTR2'))

            # Setting the critical Mach number
            f.write('{:1.3f} {:1.3f} \t\t\t |\t{} {}\n'.format(self.m_crit, self.m_ucon, 'MCRIT', 'MUCON'))

            # Setting mixed-inverse and geometry options
            f.write('{:d} {:d} \t\t\t\t\t |\t{} {}\n'.format(self.ismove, self.ispres, 'ISMOVE', 'ISPRES'))
            f.write('{:d} {:d} \t\t\t\t\t |\t{} {}\n'.format(self.n_modn, self.n_posn, 'NMODN', 'NPOSN'))

            f.close()

    def write_mses_two_element(self, airfoil_name, airfoil_state, approach='set_alpha', identifier=0):

        with open('mses.' + airfoil_name + '_' + str(identifier), 'w') as f:

            # Variables
            str_fmt = ''.join(['{:d} ' for _ in range(len(self.variables))]) + '\n'
            f.write(str_fmt.format(*self.variables))
            # Constraints
            str_fmt = ''.join(['{:d} ' for _ in range(len(self.constraints))]) + '\n'
            f.write(str_fmt.format(*self.constraints))

            # Mach number, lift coefficient & angle of attack
            if approach == 'set_alpha':
                f.write('{:1.2f} {:2.3f} {:2.3f} \t\t\t |\t{} {} {}\n'.format(airfoil_state.mach, 0.0,
                                                                              airfoil_state.alpha*180.0/np.pi,
                                                                              'MACH', 'CL', 'ALFA'))
            elif approach == 'set_cl':
                f.write('{:1.2f} {:2.3f} {:2.3f} \t\t\t |\t{} {} {}\n'.format(airfoil_state.mach, airfoil_state.c_l,
                                                                              0.0,
                                                                              'MACH', 'CL', 'ALFA'))

            # Governing equation & boundary condition
            f.write('{:d} {:d} \t\t\t\t\t |\t{} {}\n'.format(self.ismom, self.iffbc, 'ISMOM', 'IFFBC'))

            # Reynolds number & n-crit
            f.write('{:.0f} {:.1f} \t\t\t |\t{} {}\n'.format(airfoil_state.reynolds, self.n_crit, 'REYN', 'NCRIT'))

            # Transition locations
            str_fmt = ''.join(['{:1.3f} ' for _ in range(4)]) + '\t\t\t |\t{} {}' + '\n'
            f.write(str_fmt.format(self.xtr_top, self.xtr_bottom, self.xtr_top, self.xtr_bottom, 'XTR1', 'XTR2'))

            # Setting the critical Mach number
            f.write('{:1.3f} {:1.3f} \t\t\t |\t{} {}\n'.format(self.m_crit, self.m_ucon, 'MCRIT', 'MUCON'))

            # Setting mixed-inverse and geometry options
            f.write('{:d} {:d} \t\t\t\t\t |\t{} {}\n'.format(self.ismove, self.ispres, 'ISMOVE', 'ISPRES'))
            f.write('{:d} {:d} \t\t\t\t\t |\t{} {}\n'.format(self.n_modn, self.n_posn, 'NMODN', 'NPOSN'))

            f.close()

    def write_mses_input(self, airfoil_name, identifier=0):

        with open('mses_input_' + airfoil_name + '_' + str(identifier) + '.txt', 'w') as f:

            f.write('{:d}\n'.format(-10))
            f.write('{:d}\n'.format(self.n_iter))
            f.write('{:d}\n'.format(0))

            f.close()

    def write_mses_input_two_element(self, airfoil_name, identifier=0):

        with open('mses_input_' + airfoil_name + '_' + str(identifier) + '.txt', 'w') as f:

            f.write('{:d}\n'.format(self.n_iter_two_element))
            f.write('{:d}\n'.format(0))

            f.close()

    def write_mpolar(self, airfoil_name, airfoil_state, approach='set_alpha', identifier=0):

        with open('spec.' + airfoil_name + '_' + str(identifier), 'w') as f:

            if approach == 'set_alpha':
                f.write('{:d}\n'.format(5))
                f.write('{:.3f}\n'.format(airfoil_state.alpha*180.0/np.pi))
            elif approach == 'set_cl':
                f.write('{:d}\n'.format(6))
                f.write('{:.3f}\n'.format(airfoil_state.c_l))

            f.close()


    def write_mpolar_sweep(self, airfoil_name, sweep, approach='set_alpha', identifier=0):

        with open('spec.' + airfoil_name + '_' + str(identifier), 'w') as f:

            if approach == 'set_alpha':
                f.write('{:d}\n'.format(5))
                for cntr in range(len(sweep)):
                    alpha = sweep[cntr]
                    f.write('{:.3f}\n'.format(alpha*180.0/np.pi))
            elif approach == 'set_cl':
                f.write('{:d}\n'.format(6))
                for cntr in range(len(sweep)):
                    c_l = sweep[cntr]
                    f.write('{:.3f}\n'.format(c_l))

            f.close()


    def write_input(self, airfoil_name, airfoil_state, approach, identifier=0):

        # Write blade (airfoil) file
        self.write_blade(airfoil_name, airfoil=None, identifier=identifier)

        # Write mset input file
        self.write_mset_input(airfoil_name, airfoil_state, identifier=identifier)

        # Write mses file
        self.write_mses(airfoil_name, airfoil_state, approach=approach, identifier=identifier)

        # Write mses input file
        self.write_mses_input(airfoil_name, identifier=identifier)

        # Write mpolar file
        self.write_mpolar(airfoil_name, airfoil_state, approach, identifier=identifier)

    def write_input_sweep(self, airfoil_name, airfoil_state, sweep, approach, identifier=0):

        # Write blade (airfoil) file
        self.write_blade(airfoil_name, airfoil=None, identifier=identifier)

        # Write mset input file
        airfoil_state1 = airfoil_state[0]
        self.write_mset_input(airfoil_name, airfoil_state1, identifier=identifier)

        # Write mses file
        self.write_mses(airfoil_name, airfoil_state1, approach=approach, identifier=identifier)

        # Write mses input file
        self.write_mses_input(airfoil_name, identifier=identifier)

        # Write mpolar file
        self.write_mpolar_sweep(airfoil_name, sweep, approach, identifier=identifier)

    def write_input_two_element(self, airfoil_name, airfoil_state, approach, identifier=0):

        # Write blade (airfoil) file
        self.write_blade_two_element(airfoil_name, airfoil1=None, airfoil2=None, identifier=identifier)

        # Write mset input file
        self.write_mset_input_two_element(airfoil_name, airfoil_state, identifier=identifier)

        # Write mses file
        self.write_mses_two_element(airfoil_name, airfoil_state, approach=approach, identifier=identifier)

        # Write mses input file
        self.write_mses_input_two_element(airfoil_name, identifier=identifier)

        # Write mpolar file
        self.write_mpolar(airfoil_name, airfoil_state, approach, identifier=identifier)

    def run(self, airfoil_name, identifier=0):

        # Delete polar file if it exists
        if os.path.isfile('polar.' + airfoil_name + '_' + str(identifier)):
            subprocess.run(['rm ' + 'polar.' + airfoil_name + '_' + str(identifier)], shell=True)

        # Run mset
        subprocess.run([solver_settings.MSES_PATH + 'mset ' + airfoil_name + '_' + str(identifier) + ' < ' + 'mset_input_'
                        + airfoil_name + '_' + str(identifier) + '.txt' + ' > ' + 'mset_log.txt'],
                       shell=True)

        # Run mses
        subprocess.run([solver_settings.MSES_PATH + 'mses ' + airfoil_name + '_' + str(identifier) + ' < ' + 'mses_input_'
                        + airfoil_name + '_' + str(identifier) + '.txt' + ' > ' + 'mses_log.txt'],
                       shell=True)

        # Run positive mpolar sweep
        subprocess.run([solver_settings.MSES_PATH + 'mpolar ' + airfoil_name + '_' + str(identifier) + ' > ' + 'mpolar_log.txt'],
                       shell=True)

    def read_output(self, airfoil_name, identifier=0):

        # Reading positive alpha range
        rel_path = 'polar.' + airfoil_name + '_' + str(identifier)
        data = []
        if os.path.isfile(rel_path):
            data = np.loadtxt(rel_path, skiprows=13)

        if len(data) > 0:
            alpha = data[0]*(np.pi/180.0)
            c_l = data[1]
            c_d = data[2]
            c_m = data[3]
        else:
            alpha = 0.0
            c_l = -1.0
            c_d = 1.0
            c_m = 1.0

        # Output
        result = Result2(1)
        result.alpha = alpha
        result.c_l = c_l
        result.c_d = c_d
        result.c_m = c_m

        # Remove polar file
        subprocess.run(['rm -rf ' + 'polar.' + airfoil_name + '_' + str(identifier)],
                       shell=True)

        return result

    def read_output_sweep(self, airfoil_name, identifier=0):

        # Reading positive alpha range
        rel_path = 'polar.' + airfoil_name + '_' + str(identifier)
        data = []
        if os.path.isfile(rel_path):
            data = np.loadtxt(rel_path, skiprows=13)

        if len(data) > 0:
            alpha = data[:,0]#*(np.pi/180.0)
            c_l = data[:,1]
            c_d = data[:,2]
            c_m = data[:,3]
        else:
            alpha = 0.0
            c_l = -1.0
            c_d = 1.0
            c_m = 1.0

        # Output
        result = Result(len(alpha))
        result.alpha = alpha
        result.c_l = c_l
        result.c_d = c_d
        result.c_m = c_m

        # Remove polar file
        subprocess.run(['rm -rf ' + 'polar.' + airfoil_name + '_' + str(identifier)],
                       shell=True)

        return result

    def cleanup(self, airfoil_name, identifier):

        subprocess.run(['rm -rf ' + 'blade.' + airfoil_name + '_' + str(identifier)],
                       shell=True)
        subprocess.run(['rm -rf ' + 'mset_input_' + airfoil_name + '_' + str(identifier) + '.txt'],
                       shell=True)
        subprocess.run(['rm -rf ' + 'mses.' + airfoil_name + '_' + str(identifier)],
                       shell=True)
        subprocess.run(['rm -rf ' + 'mses_input_' + airfoil_name + '_' + str(identifier) + '.txt'],
                       shell=True)
        subprocess.run(['rm -rf ' + 'gridpar.' + airfoil_name + '_' + str(identifier)],
                       shell=True)
        subprocess.run(['rm -rf ' + 'spec.' + airfoil_name + '_' + str(identifier)],
                       shell=True)
        subprocess.run(['rm -rf ' + 'mdat.' + airfoil_name + '_' + str(identifier)],
                       shell=True)
        subprocess.run(['rm -rf ' + 'polarx.' + airfoil_name + '_' + str(identifier)],
                       shell=True)
        subprocess.run(['rm -rf ' + 'mset_log.txt mses_log.txt mpolar_log.txt'],
                       shell=True)


def mses_wrapper(design, state, solver, n_core, identifier=0, *args, **kwargs):

    airfoil_name = design.application_id
    if design.alpha is not None:
        mses_format = 'set_alpha'
    elif design.lift_coefficient is not None:
        mses_format = 'set_cl'
    else:
        raise Exception('mses format undefined')

    # Instantiate solver manager instance
    if design.n_crit is None:
        design.n_crit = 9.0

    mses_manager = MSES(n_crit=design.n_crit, transition_location=design.transition_location, approach=mses_format)

    # Write input
    mses_manager.write_input(airfoil_name, state, approach=mses_format, identifier=identifier)

    # Run solver
    mses_manager.run(airfoil_name, identifier=identifier)

    # Clean up files
    mses_manager.cleanup(airfoil_name, identifier=identifier)

    # Read output
    results = mses_manager.read_output(airfoil_name, identifier=identifier)

    return results

def mses_wrapper_sweep(design, state, solver, n_core, identifier=0, *args, **kwargs):

    airfoil_name = design.application_id
    if design.alpha is not None:
        mses_format = 'set_alpha'
    elif design.lift_coefficient is not None:
        mses_format = 'set_cl'
    else:
        raise Exception('mses format undefined')

    # Instantiate solver manager instance
    if design.n_crit is None:
        design.n_crit = 9.0

    mses_manager = MSES(n_crit=design.n_crit, transition_location=design.transition_location, approach=mses_format)

    # Write input
    sweep = [alpha for alpha in design.alpha]
    mses_manager.write_input_sweep(airfoil_name, state, sweep, approach=mses_format, identifier=identifier)

    # Run solver
    mses_manager.run(airfoil_name, identifier=identifier)

    # Clean up files
    mses_manager.cleanup(airfoil_name, identifier=identifier)

    # Read output
    results = mses_manager.read_output_sweep(airfoil_name, identifier=identifier)

    return results

def mses_wrapper_sweep_optimisation(design, state, solver, n_core, identifier=0, *args, **kwargs):

    airfoil_name = design.application_id
    mses_format = 'set_alpha'

    # Instantiate solver manager instance
    if design.n_crit is None:
        design.n_crit = 9.0

    mses_manager = MSES(n_crit=design.n_crit, transition_location=design.transition_location, approach=mses_format)

    # Write input
    sweep = []
    for cntr in range(len(state)):
        sweep.append(state[cntr].alpha)
    mses_manager.write_input_sweep(airfoil_name, state, sweep, approach=mses_format, identifier=identifier)

    # Run solver
    mses_manager.run(airfoil_name, identifier=identifier)

    # Clean up files
    mses_manager.cleanup(airfoil_name, identifier=identifier)

    # Read output
    results = mses_manager.read_output_sweep(airfoil_name, identifier=identifier)

    return results

def mses_wrapper_two_element(design, state, solver, n_core, identifier=0, *args, **kwargs):

    airfoil_name = design.application_id
    if design.lift_coefficient is not None:
        mses_format = 'set_cl'
    elif design.alpha is not None:
        mses_format = 'set_alpha'
    else:
        raise Exception('mses format undefined')

    # Instantiate solver manager instance
    if design.n_crit is None:
        design.n_crit = 9.0

    mses_manager = MSES(n_crit=design.n_crit, transition_location=design.transition_location, approach=mses_format)

    # Write input
    mses_manager.write_input_two_element(airfoil_name, state, approach=mses_format, identifier=identifier)

    # Run solver
    mses_manager.run(airfoil_name, identifier=identifier)

    # Clean up files
    mses_manager.cleanup(airfoil_name, identifier=identifier)

    # Read output
    results = mses_manager.read_output(airfoil_name, identifier=identifier)

    return results