import os
import numpy as np


def cleanup_quick_results():
    """
    clean up the text files with the quick results of the previous runs
    :return:
    """
    obj_file = './results/quick_obj_history.txt'
    cons_file = './results/quick_cons_history.txt'

    if os.path.isfile(obj_file):
        os.remove(obj_file)

    if os.path.isfile(cons_file):
        os.remove(cons_file)


def write_obj_cons(objective, constraint):
    """
    write the quick results files (files that get written once an individual is processed instead of at the end of
    a generation. This allows checking where the optimiser is at in case the function calls are expensive
    :param objective: contains all the objectives to be written
    :param constraint: contains all the constraints to be written
    :return:
    """

    # check if the results folder exists
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    # Write objectives
    with open('./results/quick_obj_history.txt', 'ab') as f:
        np.savetxt(f, objective[None, :], fmt='%.6f')
    # Write constraints
    with open('./results/quick_cons_history.txt', 'ab') as f:
        np.savetxt(f, constraint[None, :], fmt='%.6f')

        
def write_obj_cons_uncertainty(objective, constraint, uncertainty):
    """
    write the quick results files (files that get written once an individual is processed instead of at the end of
    a generation. This allows checking where the optimiser is at in case the function calls are expensive
    :param objective: contains all the objectives to be written
    :param constraint: contains all the constraints to be written
    :param uncertainty: contains all stat modes and Sobol sensitivities
    :param >>>
    :return:
    """

    # check if the results folder exists
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    # Write objectives
    with open('./results/quick_obj_history.txt', 'ab') as f:
        np.savetxt(f, objective[None, :], fmt='%.6f')
    # Write constraints
    with open('./results/quick_cons_history.txt', 'ab') as f:
        np.savetxt(f, constraint[None, :], fmt='%.6f')
    # Write obj stats: std, var, skew, kurt, sobol1, sobol2
    with open('./results/quick_stat_history.txt', 'ab') as f:
        np.savetxt(f, uncertainty[:], fmt='%.6f',delimiter=',')
        pass

