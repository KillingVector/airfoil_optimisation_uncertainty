from lib.airfoil_parametrisation import initialise_airfoil
from lib import flight
import numpy as np

class Design(object):
    """
    class that contains all the design related parameters.
    """

    def __init__(self, parametrisation_method, n_var, application_id, design_objectives):
        # Airfoil parameters
        self.n_pts = 201
        self.delta_z_te = 0.0
        self.parametrisation_method = parametrisation_method
        self.n_var = n_var
        self.n_crit = 9
        self.transition_location = [1.0, 1.0]
        self.airfoil = initialise_airfoil(self.parametrisation_method)
        self.shape_variables = []
        self.max_lift = False
        self.max_lift_angle = None
        self.number_of_allowed_reversals = None

        # Operating conditions
        self.application_id = application_id
        self.n_fc = 0
        self.flight_condition = []
        self.lift_coefficient = None
        self.alpha = None
        self.reynolds = None
        self.mach = None
        self.pitching_moment = None
        self.max_lift_constraint = None
        self.leading_edge_radius_constraint = None
        self.unrealistic_value = None
        self.obj_weight = None
        self.set_operating_conditions()

        # Initialise constraints
        # pitching moment and max lift constraints (if any) are part of the operating conditions for now
        self.max_thickness_constraint = None
        self.max_thickness_margin = None
        self.thickness_constraints = None
        self.area_constraint = None
        self.curvature_constraints = None
        self.curvature_variation_constraints = None
        self.set_constraints()
        self.n_con = 0

        # Initialise Objectives
        self.objective = design_objectives
        self.n_obj = len(design_objectives)

        # Initialise Uncertainty
        self.sigma_confidence = 1 # number of std devs in uncertainty bounds

    def set_operating_conditions(self):
        """
        This is where you set the operating conditions. Set up a new application_id if you want to change conditions for
        a new investigation. Set the application_id to be used in the main.py file
        :return:
        """
        # Design flight condition parameters
        if self.application_id == 'Skawinski':
            self.lift_coefficient = [0.4, 0.8, 0.85, 1.25, 1.39]
            design_re = 88.5e3 * 0.8 ** 0.5  # xfoil type 2 style ( so Re*sqrt(Cl is constant)
            self.reynolds = [design_re / c_l ** 0.5 for c_l in self.lift_coefficient]
            self.mach = [0.0256] * len(self.reynolds)
            self.obj_weight = [3.0, 10.0, 3.0, 3.0, 3.0]
            self.pitching_moment = None
            self.unrealistic_value = 60.0
            self.n_crit = 9
            self.transition_location = [0.3, 0.3]
            self.max_lift = False
            self.max_lift_constraint = self.lift_coefficient[-1]
            # based on the mh32 as this was the smallest of all the ones in Skwawinski
            self.leading_edge_radius_constraint = 0.00318
            # number of allowed reversal
            self.number_of_allowed_reversals = 2
        elif self.application_id == 'eVTOL':
            self.lift_coefficient = [0.6, 0.75, 0.85, 0.9665, 1.05, 1.25, 1.45, 1.65]
            design_re = 312.101e3 * 0.9665 ** 0.5  # xfoil type 2 style ( so Re*sqrt(Cl is constant)
            self.reynolds = [design_re / c_l ** 0.5 for c_l in self.lift_coefficient]
            self.mach = [0.08] * len(self.reynolds)
            self.obj_weight = [3.0, 6.0, 10.0, 15.0, 10.0, 10.0, 6.0, 6.0]
            self.pitching_moment = None
            self.unrealistic_value = 120.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
            self.max_lift = False
            self.max_lift_constraint = self.lift_coefficient[-1]
            # based on the SD7062
            self.leading_edge_radius_constraint = 0.0135
            # number of allowed reversal
            self.number_of_allowed_reversals = 2
        elif self.application_id == 'MAV_BWB':
            pass
        elif self.application_id == 'AMSL':
            self.lift_coefficient = [0.5, 0.7, 0.9, 0.9]
            design_re = 2.4e6 * 0.7 ** 0.5  # xfoil type 2 style ( so Re*sqrt(Cl is constant)
            self.reynolds = [design_re / c_l ** 0.5 for c_l in self.lift_coefficient]
            self.mach = [0.0] * len(self.reynolds)
            self.obj_weight = [3.0, 10.0, 3.0, 0.0]
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [0.3, 0.3]
        elif self.application_id == 'prop_0775r':
            # conditions are cruise a, cruise b, cruise c, hover a, hover b, hover f, manoeuvre
            # hover c not included as it is very close to hover b
            # cruise f not included as it is very close to cruise c
                # self.lift_coefficient = [0.2441, 0.3096, 0.3823, 0.7698, 0.8475,  1.1326]
            # self.reynolds = [644666, 828326, 857446, 895926, 863370, 780992]
            # self.mach = [0.5915, 0.6396, 0.6844, 0.6849, 0.6889, 0.6869]

            self.lift_coefficient = [0.10, 0.14, 0.26, 0.70, 0.85, 1.20, 1.22]
            self.reynolds = [1050000, 1000000, 1010000, 1050000, 950000, 1200000, 1300000]
            self.mach = [0.57, 0.51, 0.40, 0.53, 0.47, 0.47, 0.53]
            # MH120 - 11.6% / MH121 - 8.8%

            self.alpha = [0.0, 1.0, 2.0, 8.0, 10.0, 10.0]
            self.alpha = [a*(np.pi/180) for a in self.alpha]

            # remove maneuver point from here on
            self.obj_weight = None
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
            # based on the naca4412
            # self.leading_edge_radius_constraint = 0.016

        elif self.application_id == 'prop_0775r_thin':
            # conditions are cruise a, cruise b, cruise c, hover a, hover b, hover f, manoeuvre
            # hover c not included as it is very close to hover b
            # cruise f not included as it is very close to cruise c
            # self.lift_coefficient = [0.2441, 0.3823, 0.3096, 0.7698, 0.8475, 1.1326, 1.6326]
            # self.reynolds = [644666, 857446, 828326,  895926, 863370, 780992, 780992]
            # self.mach = [0.5915, 0.6844, 0.6396, 0.6869, 0.6849, 0.6889, 0.6]

            self.lift_coefficient = [0.10, 0.14, 0.26, 0.70, 0.85, 1.20, 1.22]
            self.reynolds = [1050000, 1000000, 1010000, 1050000, 950000, 1200000, 1300000]
            self.mach = [0.57, 0.51, 0.40, 0.53, 0.47, 0.47, 0.53]

            self.alpha = [0.0, 2.0, 1.0, 8.0, 10.0, 10.0]
            self.alpha = [a*(np.pi/180) for a in self.alpha]
            self.obj_weight = None
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
            # based on the naca4409
            # self.leading_edge_radius_constraint = 0.0093
        elif self.application_id == 'prop_0995r':
            # conditions are cruise a, cruise b, cruise c, hover a, hover b, hover f, manoeuvre
            # hover c not included as it is very close to hover b
            # cruise f not included as it is very close to cruise c
            # self.lift_coefficient = [0.0902, 0.4010, 0.2500, 0.8550, 0.6290, 0.6752, 1.355]
            # self.reynolds = [546325, 739090, 691138, 684037, 791372, 671282, 684037]
            # self.mach = [0.7065, 0.8237, 0.7672, 0.8475, 0.8455, 0.8495, 0.8]
            self.lift_coefficient = [0.0, 0.1, 0.4, 0.65, 0.82, 1.1]
            self.reynolds = [900000, 1400000, 1300000, 1100000, 1225000, 1300000]
            self.mach = [0.50, 0.69, 0.65, 0.58, 0.66, 0.59]
            self.alpha = [0.0, 2.0, 1.0, 6.0, 8.0, 10.0]
            self.alpha = [a*(np.pi/180) for a in self.alpha]
            self.obj_weight = None
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
        elif self.application_id == 'prop_0505r':
            self.lift_coefficient = [0.0, 0.13, 0.25, 1.0, 1.1, 1.2]
            self.reynolds = [950000, 1000000, 900000, 950000, 700000, 800000]
            self.mach = [0.42, 0.47, 0.34, 0.40, 0.275, 0.34]
            self.alpha = [0.0, 2.0, -1.0, 10.0, 10.0, 10.0]
            self.alpha = [a*(np.pi/180) for a in self.alpha]
            self.obj_weight = None
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
        elif self.application_id == 'prop_0505r_thick':
            self.lift_coefficient = [0.0, 0.13, 0.25, 1.0, 1.1, 1.2]
            self.reynolds = [950000, 1000000, 900000, 950000, 700000, 800000]
            self.mach = [0.42, 0.47, 0.34, 0.40, 0.275, 0.34]
            self.alpha = [0.0, 2.0, -1.0, 10.0, 10.0, 10.0]
            self.alpha = [a*(np.pi/180) for a in self.alpha]
            self.obj_weight = None
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
            self.transition_location = [1.0, 1.0]
        elif self.application_id == 'prop_0235r':
            # conditions are cruise a, cruise b, cruise c, hover a, hover b, hover c, manoeuvre
            # hover c not included as it is very close to hover b
            # cruise f not included as it is very close to cruise c
            # self.lift_coefficient = [-0.5224, -0.3730, -0.6722, 1.6630, 1.5978, 1.3453]
            # self.reynolds = [361916, 536686, 599850, 251870, 353840, 408010]
            # self.mach = [0.2972, 0.3152, 0.3072, 0.1982, 0.1992, 0.2002]
            # self.alpha = [0.0, 2.0, -1.0, 10.0, 10.0, 10.0]

            self.lift_coefficient = [0.0, 0.13, 0.25, 1.0, 1.1, 1.2]
            self.reynolds = [950000, 1000000, 900000, 950000, 700000, 800000]
            self.mach = [0.42, 0.47, 0.34, 0.40, 0.275, 0.34]
            self.alpha = [0.0, 2.0, -1.0, 10.0, 10.0, 10.0]
            self.alpha = [a*(np.pi/180) for a in self.alpha]
            self.obj_weight = None
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
        elif self.application_id == 'prop_0235r_thin':
            # conditions are cruise a, cruise b, cruise c, hover a, hover b, hover c, manoeuvre
            # hover c not included as it is very close to hover b
            # cruise f not included as it is very close to cruise c
            self.lift_coefficient = [0.0, 0.13, 0.25, 1.0, 1.1, 1.2]
            self.reynolds = [950000, 1000000, 900000, 950000, 700000, 800000]
            self.mach = [0.42, 0.47, 0.34, 0.40, 0.275, 0.34]
            self.alpha = [0.0, 2.0, -1.0, 10.0, 10.0, 10.0]
            self.alpha = [a*(np.pi/180) for a in self.alpha]
            self.obj_weight = None
            self.pitching_moment = None
            self.unrealistic_value = 200.0
            self.n_crit = 9
            self.transition_location = [1.0, 1.0]
        elif self.application_id == 'airfoil_analysis':
            self.alpha = [0]
            self.lift_coefficient = None
            self.reynolds = [0]
            self.mach = [0]
            pass
        ## uncertainty testers
        elif self.application_id in ['utest','utest_unc','utest_unc2']:
            # self.lift_coefficient   = [0.4, 0.8]
            # self.reynolds           = [5e5, 2e6]
            # self.mach               = [0.2, 0.3]  # [0.1, 0.1]
            self.lift_coefficient   = [0.4, 0.8, 0.85, 1.25]
            design_re               = 1e6 * 0.8 ** 0.5  # xfoil type 2 style ( so Re*sqrt(Cl is constant)
            self.reynolds           = [design_re / c_l ** 0.5 for c_l in self.lift_coefficient]
            design_ma               = 0.2 * 0.8
            self.mach               = [design_ma / c_l ** 0.5 for c_l in self.lift_coefficient]
            self.alpha              = [0.0, 0.0]
            self.alpha              = [a*(np.pi/180) for a in self.alpha]
            self.obj_weight         = None
            self.pitching_moment    = None
            self.unrealistic_value  = 200.0
            self.n_crit             = 9
            self.transition_location= [1.0, 1.0]
            self.uncertainty_weight = 1.0
        else:
            raise Exception('This case is not implemented yet')

        # Number of flight conditions
        if self.lift_coefficient is not None:
            self.n_fc = len(self.lift_coefficient)
        elif self.alpha is not None:
            self.n_fc = len(self.alpha)
        else:
            raise Exception('You need to set the operating conditions')

        # Design flight conditions
        self.flight_condition = [flight.FlightCondition() for _ in range(self.n_fc)]
        for i, re in enumerate(self.reynolds):
            self.flight_condition[i].set(h=0.0, reynolds=re, mach=self.mach[i])
            if self.lift_coefficient is not None:
                self.flight_condition[i].c_l = self.lift_coefficient[i]
            elif self.alpha is not None:
                self.flight_condition[i].alpha = self.alpha[i]

    def set_constraints(self):

        if self.application_id == 'prop_0775r':
            self.max_thickness_constraint = 0.12
            self.max_thickness_margin = 2 # cut it off at double the thickness
        elif self.application_id == 'prop_0775r_thin':
            self.max_thickness_constraint = 0.09
            self.max_thickness_margin = 2 # cut it off at double the thickness
        elif self.application_id == 'prop_0995r':
            self.max_thickness_constraint = 0.09
            self.max_thickness_margin = 2 # cut it off at double the thickness
        elif self.application_id == 'prop_0235r':
            self.max_thickness_constraint = 0.18
            self.max_thickness_margin = 2 # cut it off at double the thickness
        elif self.application_id == 'prop_0235r_thin':
            self.max_thickness_constraint = 0.15
            self.max_thickness_margin = 2 # cut it off at double the thickness
        elif self.application_id == 'prop_0505r':
            self.max_thickness_constraint = 0.12
            self.max_thickness_margin = 2 # cut it off at double the thickness
        elif self.application_id == 'prop_0505r_thick':
            self.max_thickness_constraint = 0.15
            self.max_thickness_margin = 2 # cut it off at double the thickness
        elif self.application_id == 'eVTOL':
            self.max_thickness_constraint = 0.12
            self.max_thickness_margin = 2  # cut it off at double the thickness
        elif self.application_id in ['utest','utest_unc','utest_unc2']:
            self.max_thickness_constraint = 0.12
            self.max_thickness_margin = 2  # cut it off at double the thickness
        else:
            self.max_thickness_constraint = 0.09
            self.max_thickness_margin = 2 # cut it off at double the thickness

        if self.application_id == 'eVTOL':
            self.thickness_constraints = [[0.7, 0.8, 0.9, 0.95],  # chord,wise locations
                                          [0.03, 0.02, 0.01, 0.005]]  # actual constraints
            # self.thickness_constraints = [[0.005, 0.01, 0.05, 0.7, 0.8, 0.9, 0.95],  # chord,wise locations
            #                               [0.01, 0.02, 0.045, 0.03, 0.02, 0.01, 0.005]]  # actual constraints

        else:
            self.thickness_constraints = [[0.005, 0.01, 0.05, 0.7, 0.8, 0.9, 0.95],  # chord,wise locations
                                          [0.0135, 0.02, 0.0445, 0.03, 0.02, 0.01, 0.005]]  # actual constraints

        # self.thickness_constraints = [[0.7, 0.9, 0.95],  # chord,wise locations
        #                               [0.03, 0.01, 0.005]]  # actual constraints
        # if self.application_id == 'Skawinski':
        #     self.thickness_constraints[1] = [x * self.max_thickness_constraint / 0.09 for x in
        #                                      self.thickness_constraints[1]]
                                            # scale thickness constraints for fatter airfoils
        self.thickness_constraints[1] = [x * self.max_thickness_constraint / 0.09 for x in
                                         self.thickness_constraints[1]]

        # Note: pitching moment constraints are set in set_operating_conditions - NOT HERE
        # based on AG26 (lowest area of all the ones in Skawinski)
        if self.application_id == 'eVTOL':
            self.area_constraint = 0.075
            # self.area_constraint = None
        if self.application_id == 'utest':
            self.area_constraint = None
        elif self.application_id == 'prop_0775r':
            self.area_constraint = None
        elif self.application_id == 'prop_0775r_thin':
            # self.area_constraint = 0.0616
            self.area_constraint = None
        else:
            self.area_constraint = None

        self.curvature_constraints = None
        self.curvature_variation_constraints = None

        # # constraints as taken from matlab code
        # self.curvature_constraints = [[0.2, 0.8], # chord-wise start and end
        #                               [2, 10], # curvature restrictions
        #                               [30, 100]] # radius of curvature restrictions (smoothness)

        # tighter constraints
        # self.curvature_constraints = [[0.2, 0.9], # chord-wise start and end
        #                               [1, 5], # curvature restrictions
        #                               [2, 10]] # radius of curvature restrictions (smoothness)

        # # tightest constraints - as proposed by Wang (for much thicker airfoil though
        # self.curvature_constraints = [[0.2, 0.8], # chord-wise start and end
        #                               [0.5, 20], # curvature restrictions
        # self.curvature_variation_constraints = [[0.2, 0.8], # chord-wise start and end
        #                               [30, 100]] # radius of curvature restrictions (smoothness)
        # TODO: add them in with different sign for top and bottom as done in Wang2016aaa
