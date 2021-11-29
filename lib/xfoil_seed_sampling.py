import numpy as np
import pickle
import os
import pickle5
import subprocess
import random
from pyDOE2 import lhs

from optimisation.model.sampling import Sampling
from lib.utils import generate_variables_from_xfoilrun, airfoil_analysis_xfoil

def initialise_airfoils_with_xfoil(case_name, opt_prob, design):
    # read in shape variables from xfoil run
    src= './results/optimisation_history_' + case_name + '.pkl'
    filename = './lib/normalised_seeds_'+ case_name + '.pkl'
    filename2 = './lib/normalised_seeds.pkl'
    if not os.path.isfile(filename):
        values = generate_variables_from_xfoilrun(src)
        # rerun them through xfoil as the angles need to be added
        save_variable = []
        for idx in range(len(values)):
            shape_vars = values[idx]
            angles = airfoil_analysis_xfoil(design, shape_vars, idx)
            tempdata = np.hstack((shape_vars, angles*(np.pi/180.0)))
            save_variable = np.append(save_variable, tempdata, axis=0)
        # now reshape
        save_variable2 = np.reshape(save_variable,(len(values),-1))
        save_variable2 = (save_variable2 - opt_prob.x_lower)/(opt_prob.x_upper - opt_prob.x_lower)
        with open(filename, 'wb') as f:
            pickle5.dump(save_variable2, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    cmd = 'cp "%s" "%s"' % (filename, filename2)
    status = subprocess.call(cmd, shell=True)