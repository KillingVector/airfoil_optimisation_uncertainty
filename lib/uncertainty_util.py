#import matplotlib.pyplot as plt
import numpy as np
import uncertainpy as un
import chaospy as cp
import os
#from sklearn.linear_model import LinearRegression




def create_sample(variable = None, var_dist = 0.1, initial_dist = 'normal', order = 2):#, rule = "Gaussian", order = 2):
    # TODO: intitial_dist must deal with list
    if not variable:
        raise Exception("No variables assigned")

    # TODO: NOTE: input std != output std for initial distributions
    if initial_dist.lower() in ['normal','norm','standard']:
        # adjust = 0.3066*order + 0.7574
        p = np.array([ 0.02711111, -0.28161905,  1.17765079,  0.01838095])
        adjust = np.polyval(p,order)
        if not type(var_dist) == type(list()):
            var_dist = var_dist / adjust
        else:
            for i,v in enumerate(var_dist):
                var_dist[i] = v / adjust
    
    # only does 2 variables at the moment
    if initial_dist.lower() in ['normal','norm','standard']:
        if not type(var_dist) == type(list()):
            A = cp.Normal(variable[0], var_dist*variable[0])
            B = cp.Normal(variable[1], var_dist*variable[1])
        else:
            A = cp.Normal(variable[0], var_dist[0]*variable[0])
            B = cp.Normal(variable[1], var_dist[1]*variable[1])
    elif initial_dist.lower() in ['uniform','linear']:
        if not type(var_dist) == type(list()):
            A = cp.Uniform(variable[0]-var_dist*variable[0], 
                            variable[0]+var_dist*variable[0])
            B = cp.Uniform(variable[1]-var_dist*variable[1], 
                            variable[1]+var_dist*variable[1])
        else:
            A = cp.Uniform(variable[0]-var_dist[0]*variable[0],        
                            variable[0]+var_dist[0]*variable[0])
            B = cp.Uniform(variable[1]-var_dist[1]*variable[1], 
                            variable[1]+var_dist[1]*variable[1])
    # only does 2 variables at the moment

    dist_list = [A, B]
    return dist_list
    
def generate_quadrature(dist_list = None, rule = "Gaussian", order = 2):
    # Only does 2 var at the moment
    A = dist_list[0]
    B = dist_list[1]
    joint    = cp.J(A,B)
    nodes, weights = cp.generate_quadrature(order = order, dist = joint, rule = rule)
    
    return nodes, weights, joint

## dependent samples are created by multiplying distributions together

def get_evals(solver = None, nodes = None): # model can be passed in this way
    
    if not solver:
        raise Exception("Require solver to return evaluations")
    if type(nodes) == type(None):
        raise Exception("No sample points for model provided")

    evals = [solver(node) for node in nodes.T]
    return evals # don't need to use this function, just need evals in a list


    
def create_model(nodes = None, weights = None, evals = None, joint = None, order = 2):#, dependent = False):

    """ ---> text nicked from uncertainpy
            The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.
    """
    
    if type(nodes) == type(weights) == type(evals) == type(joint) == type(None):
        raise Exception("Nodes, Weights, Evals and Joint cannot be NoneType")

    # TODO make sure that evals is in correct shaped list format
    if len(weights) == 0 or not type(weights) == type(list()):
        expansion   = cp.generate_expansion(order, joint)
        try:
            model       = cp.fit_quadrature(expansion, nodes, weights, evals)
        except Exception as e:
            model = None
    else:
        expansion   = None
        model       = None
    return model


def check_failures(evals = None, FAIL_CRITERIA = -1):
    ## TODO at the moment you need to manually adjust your failure criteria
    fail    = []
    for i in range(0,len(evals)):
        # failure criteria here
        if not type(evals[i]) == type(list()):
            if not type(evals[i]) == type(None):
                if evals[i] <= FAIL_CRITERIA:
                    fail.append(i)
                elif evals[i] == np.nan:
                    fail.append(i)
        else: # if list, check main criteria
            if not type(evals[i]) == type(None):
                if evals[i][0] <= FAIL_CRITERIA:
                    fail.append(i)
                elif evals[i][0] == np.nan:
                    fail.append(i)
        # other failure criteria here
    return fail

def correct_nodes_weights(nodes = None, weights = None, fail = []):
    # remove failed runs
    for i in reversed(fail):
        nodes   = np.delete(nodes, i, 1)
        weights = np.delete(weights, i, 0)
    return nodes, weights

def correct_evals(evals = None, fail = []):
    # remove failed runs
    for i in reversed(fail):
        evals   = np.delete(evals, i)
    evals = np.array(evals)
    return evals


def get_statistics(model = None, distribution = None):

    '''
        distribution -> the distribution upon which the pce solution was created
    '''
    # skewness, especially for environmental errors, will show if there is particular variance in data weights over error bounds - eg approaching transonic region
    # skewness will also indicate if confidence intervals/std deviation ranges, if the skewness is very large
    
    #kurtosis, does it look more like a delta function or a box? Where 3 is a normal distribution
    # -- sort of relates to concavity, <3 mostly convex, >3 mostly concave
    # <3 means there are fewer outliers, data doesn't taper at extremities
    # >3 means that the data has a high central concentration, but also strong outlier presence
    # kurtosis can indicate the difference between a cosine distribution (<3) and a normal (3) - useful in predicting x-sigma requirements for capturing all possible data.

    try:
        mean    = np.float64(cp.E(model, distribution))
        std     = cp.Std(model, distribution)
        variance= np.float64(cp.Var(model, distribution))
        sobol   = cp.Sens_m(model, distribution) # there are three sobol options

        skew    = np.float64(cp.Skew(model, distribution))
        kurt    = np.float64(cp.Kurt(model, distribution))

        stats1 = [mean, std, variance, skew, kurt]
        stats = [stats1, sobol]
    except Exception as e:
        # this is generally used for C_l in sweep simulations, as the same C_l is used,
        # and thus there are nan and zero stat values
        sobol = [-1, -1]
        stats1 = [-1, -1, -1, -1, -1]
        stats = [stats1, sobol]
    return stats
    
# TODO move to sim_unc.py
def run_uncertainty(design, config):
    for fc_idx in range(design.n_fc):
        # TODO uncertainty setup here - may need calculations
        # uncertainty_util imported as ut
        unc     = config.settings.uncertainty
        if unc.tag.lower() == 'vt':
            if design.flight_condition[fc_idx].u == 0.0 or design.flight_condition[fc_idx].ambient.T == 0.0:
                design.flight_condition[fc_idx].MRe_to_velT()
            vel = design.flight_condition[fc_idx].u
            T   = design.flight_condition[fc_idx].ambient.T
            var = [vel, T]
        elif unc.tag.lower() == 'mre':
            mach= design.flight_condition[fc_idx].mach
            re  = design.flight_condition[fc_idx].reynolds
            var = [mach, re]
        # create nodes
#                nodes, weights, dist    = ut.create_sample(variable = var,
#                                                        var_dist    = unc.dist,
#                                                        initial_dist= unc.initial,
#                                                        rule        = unc.rule,
#                                                        order       = unc.order)
        dist_list           = create_sample(variable = var,
                                                var_dist    = unc.dist,
                                                initial_dist= unc.initial)
        nodes, weights, dist= generate_quadrature(dist_list = dist_list,
                                                    rule        = unc.rule,
                                                    order       = unc.order)
        return nodes, weights, dist



    
        

'''

======================   TEST AREA ==============================

'''
    
def mod1(parameters):
    vel = parameters[0]
    T   = parameters[1]

    R   = 287
    p   = 101.3e3
    L   = 1
    mu  = 1.789e-5
    gamma=1.4
    mass= 20
    W   = 9.81 * mass
    S   = 4.5
    rho = p / R / T
    return W / (0.5 * rho * vel**2 * S)
    
def equiv(nodes):

    vel = nodes[0,:]
    T   = nodes[1,:]

    M   = [ vel[i] / (1.4 * 287 * T[i] ) ** (1/2) for i in range(0,len(vel)) ]
    Re  = [ (101.3e3 / (287 * T[i] ) ) * ( vel[i] * 1.0 ) / ( 1.789e-5) for i in range (0,len(vel)) ]

#    print(np.mean(Re))
#    print(np.std(Re))
#    print(np.std(Re)/np.mean(Re))
#    quit()
    mach    = [ np.mean(M), np.std(M)/np.mean(M) ]
    reynolds= [ np.mean(Re), np.std(Re)/np.mean(Re) ]

    return mach, reynolds
    

if __name__ == '__main__':


    dist_list = create_sample([0.1, 2.3e5], 0.1, 'normal')
    _, _, joint = generate_quadrature(dist_list=dist_list, rule="Gaussian", order=2)

    nodes = np.array([[8.26794919e-02, 8.26794919e-02, 1.00000000e-01, 1.00000000e-01,
        1.00000000e-01, 1.17320508e-01, 1.17320508e-01, 1.17320508e-01],
       [2.30000000e+06, 2.69837169e+06, 1.90162831e+06, 2.30000000e+06,
        2.69837169e+06, 1.90162831e+06, 2.30000000e+06, 2.69837169e+06]])
    weights = np.array([0.11111111, 0.02777778, 0.11111111, 0.44444444, 0.11111111,
       0.02777778, 0.11111111, 0.02777778])

    evals = np.array([0.3346, 0.3353, 0.3344, 0.3351, 0.3359, 0.3351, 0.3359, 0.3367])

    model = create_model(nodes = nodes, weights = weights, evals = evals, joint = joint, order = 2)

    quit()


    var = [30, 288.15] # 136 m/s, 288K and Mach 0.4, 2.9e6 Re
    var_dist =  0.1
    initial_dist = 'normal'
    rule = 'Gaussian'
    order = 2# order for dist and polyfit MUST be equal - tested 
    x = var[0]
    y = var[1]
    
    dist_list = create_sample(variable=var,
                                var_dist = var_dist,
                                initial_dist = initial_dist)
    nodes, weights, joint = generate_quadrature(dist_list = dist_list,
                                                rule = rule,
                                                order = order)
                                        
    mach, reynolds = equiv(nodes)

    # instead of get_evals, you can just manually call in the code
    # only needs to output list
    evals = get_evals(solver = mod1, nodes = nodes)
    # comment these in to get 'failed models'
#    evals[1] = -13
#    evals[5] = -15
#    evals[7] = -20
    # comment in below to check for multiple evaluation results eg [L/D, mass]
#    oldevals = evals
#    evals   = []
#    for ev in oldevals:
#        evals.append([ev, ev])

    # nodes, weights, evals = check_failures(evals = evals,
    #                                        FAIL_CRITERIA = -1)

    [print(evals[i]) for i in range(0,len(evals))]
    model   = create_model(nodes    = nodes, 
                            weights = weights, 
                            evals   = evals, 
                            joint   = joint,
                            order   = order)


    stats, _    = get_statistics(model     = model, distribution   = joint)#,
#                                      model2    = model2, distribution2 = dist2)    
    
    
    [mean, std, variance, skew, kurt] = stats[0]
    sens    = stats[1]
    pcstd   = std/mean
    print("mean",mean)
    print("stats",stats[0])
    print("std type", type(std))


    
    
