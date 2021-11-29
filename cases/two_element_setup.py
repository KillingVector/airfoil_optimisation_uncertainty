import numpy as np

from lib import config, utils
from lib import simulation_two_element as simulation
from optimisation.setup import Setup


class TwoElementSetup(Setup):

    def __init__(self):
        super().__init__()

    def set_variables(self, prob):
        """
        add the variables to the instance of the problem class
        :param prob: instance of the problem class (created in main.py)
        :return: None
        """
        # this is very specific for this case as it uses a list of fitted airfoils to get the solver started off right
        # would not be here in a generic case where initial sampling is used
        lower_bound, upper_bound, initial_guess = utils.set_coefficient_bounds(config.design.parametrisation_method,
                                                                               config.design.n_var)
        # add the variables that determine the shape of the airfoil (so the parametrisation related variables)
        # for the main element
        # prob.add_var_group('shape_vars', config.design.n_var, 'c',
        #                    lower=lower_bound.tolist(),
        #                    upper=upper_bound.tolist(),
        #                    value=initial_guess.tolist(), scale=1.0)
        # # for the second element
        # prob.add_var_group('shape_vars_element2', config.design.n_var, 'c',
        #                    lower=lower_bound.tolist(),
        #                    upper=upper_bound.tolist(),
        #                    value=initial_guess.tolist(), scale=1.0)

        # scale of the elements
        prob.add_var_group('scale_element1', 1, 'c',
                           lower=0.7, upper=0.9,
                           value=0.8, scale=2.0)
        prob.add_var_group('scale_element2', 1, 'c',
                           lower=0.1, upper=0.5,
                           value=0.3, scale=2.0)

        # rotation angle for the second element
        prob.add_var_group('rotation_angle_element2', 1, 'c',
                           lower=-5.0 * (np.pi / 180.0), upper=15.0 * (np.pi / 180.0),
                           value=5.0 * (np.pi / 180.0), scale=2.0)

        # translation for element 2 (x and z)
        prob.add_var_group('translation_element2', 2, 'c',
                           lower=0.0, upper=-0.2,
                           value=-0.1, scale=2.0)


        # if you are using the alpha option in xfoil and MSES or you are running CFD you need to set an angle of attack
        if (config.design.lift_coefficient is not None and config.settings.solver in ['su2', 'openfoam']) \
                or config.design.alpha is not None:
            prob.add_var_group('angle_of_attack', config.design.n_fc, 'c',
                               lower=-5.0*(np.pi/180.0), upper=15.0*(np.pi/180.0),
                               value=0.0*(np.pi/180.0), scale=2.0)
        if config.design.max_lift:
            prob.add_var_group('max_lift_angle', 1, 'c',
                               lower=-5.0*(np.pi/180.0), upper=15.0*(np.pi/180.0),
                               value=0.0*(np.pi/180.0), scale=2.0)

    def set_constraints(self, prob, **kwargs):
        """
        set all the constraints for the case
        :param prob: instance of the problem class (created in main.py)
        :return: None
        """
        # constraint 0 and 1 are cross-over
        prob.add_con_group('cross_over', 2, lower=None, upper=0.0)
        # constraint 2 checks if the elements are over-lapping
        prob.add_con_group('element_cross_over', 1, lower=None, upper=0.0)
        # constraint 3 and 4 are max thickness
        if config.design.max_thickness_constraint is not None:
            prob.add_con_group('max_thickness', 2, lower=None, upper=0.0)
        # constraint 5 to 5+2*n_thickness_constraints is thickness
        if config.design.thickness_constraints is not None:
            prob.add_con_group('thickness', 2*len(config.design.thickness_constraints[0]), lower=None, upper=0.0)
        if config.design.number_of_allowed_reversals is not None:
            prob.add_con_group('reversals', 2, lower=None, upper=0.0)
        # constraints to ensure that it meets the lift requirements
        prob.add_con_group('minimum_lift', config.design.n_fc, lower=None, upper=0.0)
        #  pitching moment and others if they are specified
        if config.design.pitching_moment is not None:
            prob.add_con_group('pitching_moment',  len(config.design.pitching_moment[0]), lower=None, upper=0.0)
        if config.design.area_constraint is not None:
            prob.add_con_group('area', 2, lower=None, upper=0.0)
        if config.design.leading_edge_radius_constraint is not None:
            prob.add_con_group('le_radius', 2, lower=None, upper=0.0)
        if config.design.curvature_constraints is not None:
            prob.add_con_group('curvature', 8, lower=None, upper=0.0)
        if config.design.curvature_variation_constraints is not None:
            prob.add_con_group('curvature_variation', 8, lower=None, upper=0.0)
        if config.design.max_lift:
            # constraints to ensure that it meets the lift requirements at max lift condition
            prob.add_con_group('meets_max_lift', 1, lower=None, upper=0.0)



    def set_objectives(self, prob, **kwargs):
        """
        add the objectives to the problem class - here they are just generic names. The actual objective "type" is set
        in main.py (less generic but easier to change)
        :param prob: instance of the problem class (created in main.py)
        :return: None
        """

        if config.design.n_obj == 1:
            prob.add_obj('f_1')
        elif config.design.n_obj == 2:
            prob.add_obj('f_1')
            prob.add_obj('f_2')
        elif config.design.n_obj == 3:
            prob.add_obj('f_1')
            prob.add_obj('f_2')
            prob.add_obj('f_3')
        else:
            raise Exception('So far only 1, 2 or 3 objectives has been set up')

    def obj_func(self, x_dict, idx=0, **kwargs):
        """
        define the objective function
        :param x_dict: dictionary containing all the design variable values
        :param idx: identifier for the current solution (to prevent contamination if run in parallel)
        :param kwargs: other keyword arguments as needed for the specific solver
        :return: obj, cons, performance: objectives, constraints, and a performance parameter for analysis
        """
        # Resetting design viability flag
        config.design.viable = True
        lower_bound, upper_bound, initial_guess = utils.set_coefficient_bounds(config.design.parametrisation_method,
                                                                               config.design.n_var)

        # Shape variables
        # config.design.shape_variables = x_dict['shape_vars']
        # config.design.shape_variables_element2 = x_dict['shape_vars_element2']
        config.design.shape_variables = initial_guess
        config.design.shape_variables_element2 = initial_guess
        config.design.rotation_element2 = x_dict['rotation_angle_element2']
        config.design.translation_element2 = x_dict['translation_element2']
        config.design.scale_element1 = x_dict['scale_element1']
        config.design.scale_element2 = x_dict['scale_element2']

        # Angle of attack
        if 'angle_of_attack' in x_dict.keys():
            aoa = x_dict['angle_of_attack']
            for i in range(len(aoa)):
                config.design.flight_condition[i].alpha = aoa[i]
        if 'max_lift_angle' in x_dict.keys():
            config.design.max_lift_angle = x_dict['max_lift_angle']

        # Evaluating objective function - calls simulation.py which is where you set up the actual simulation for the
        # airfoil
        obj, cons, performance = simulation.airfoil_analysis(config.design, idx, **kwargs)
        return obj, cons, performance

    def obj_func_specific(self, design, idx=0, **kwargs):

        # Evaluating objective function
        obj, _, _ = simulation.airfoil_analysis(design, idx, **kwargs)

        return obj

    def cons_func_specific(self, design, idx=0, **kwargs):

        # Evaluating objective function
        _, cons, _ = simulation.airfoil_analysis(design, idx, **kwargs)

        return cons