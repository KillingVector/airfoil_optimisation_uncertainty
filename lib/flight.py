import numpy as np

# TODO change it over to formulation of ADRpy?


class Viscosity(object):
    def __init__(self):

        self.mu = 0.0   # Dynamic viscosity [N s/m**2.0]
        self.nu = 0.0   # Kinematic viscosity [m**2.0/s]
        self.mu_0 = 0.0
        self.mu_T = 0.0
        self.sutherland = 0.0 # Sutherland's constant


class FlightCondition(object):
    def __init__(self):

        # Altitude [m]
        self.h = 0.0
        self.atmosphere = 'earth'

        # Ambient conditions
        if self.atmosphere in ['mars','martian','co2','carbondioxide','carbon dioxide']:
            self.ambient = Environment(gas = 'co2')
        else:
            self.ambient = Environment() # Earth, air default

        # Flight velocity [m/s]
        self.u = 0.0

        # Reynolds number
        self.reynolds = 0.0
        self.Re       = 0.0 #ugh

        # Mach number
        self.mach = 0.0

        # Angle of attack [rad]
        self.alpha = 0.0
        
        # Simulation conditions
        self.compressible       = True
        self.TurbulenceIntensity= 0.05
        self.chord              = 1.0 # as reynolds length for dim sims
        self.y_plus             = 1.0

        # Aerodynamic conditions
        self.c_l = 0.0
        self.c_d = 0.0
        self.c_m = 0.0

    def set(self, h=0.0, u=0.0, reynolds=0.0, mach=0.0, alpha=None):

        # Set parameters
        self.h = h
        self.u = u
        self.reynolds = reynolds
        self.mach = mach
        self.alpha = alpha

        # Calculate ambient conditions for current flight condition
        # if gravity is earthlike
        if self.atmosphere in [None, 'air', 'earth'] and not self.h == None:
            self.ambient.isa(self.h)
        else: # use custom atmosphere settings
            self.ambient.custom()
        
        # Calculate velocity if not given
        if u in [None, 0.0, 0]:
            self.u  = self.mach * self.ambient.a
    
    # convert mach+reynolds to velocity+Temp            
    def MRe_to_velT(self, recreate = True):
        run = False
        L_ref = self.chord
        if self.ambient.T == 0.0 or self.ambient.T == None or recreate:
            run = True

        if run:
            self.u  = L_ref*self.mach**2 * self.ambient.gamma * self.ambient.P / (self.reynolds * self.ambient.viscosity.mu)
            self.ambient.T  = L_ref**2 * self.mach**2 * self.ambient.gamma * self.ambient.P**2 / (self.ambient.R * self.reynolds**2 * self.ambient.viscosity.mu**2)
        # lossy calc due to large values, apply rounding
        self.u          = np.round(self.u,3)
        self.ambient.T  = np.round(self.ambient.T,3)
    # convert velocity+Temp to mach+reynolds        
    def velT_to_MRe(self):
        L_ref = self.chord
        self.reynolds   = self.ambient.P / (self.ambient.R * self.ambient.T) * self.u * L_ref / self.ambient.viscosity.mu
        self.mach       = self.u/(self.ambient.gamma * self.ambient.R * self.ambient.T)**0.5
        # lossy calc due to large values, apply rounding
        self.reynolds   = np.round(self.reynolds,6)
        self.mach       = np.round(self.mach,6)
    def regen(self):
        # this may need to be done for cfd cases - must check with Dries
        # http://www.ae.utexas.edu/~varghesep/class/propulsion/gamma_air.GIF
        pass
    # ordered calls of these values - for use with uncertainty node evals
    def vm_set(self):
        self.velT_to_MRe()
        self.MRe_to_velT()
    def mv_set(self):
        self.MRe_to_velT()
        self.velT_to_MRe()



        


class Environment(object):
    def __init__(self, gas = 'air'):

        self.gas = gas  # air, co2
        self.rho = 0.0  # Air density [kg/m**3.0]
        self.T = 0.0    # Temperature [K]
        self.P = 0.0    # Pressure [Pa]
        self.viscosity = Viscosity()
        self.a = 0.0    # Speed of sound [m/s]

    def isa(self, altitude):

        # Constants
        g_0 = 9.80665   # Acceleration due to gravity [m/s**2.0]
        rho_0 = 1.2256
        T_0 = 288.15
        P_0 = 101325
        
        self.g  = g_0 # save gravity

        # Viscosity
        mu_0 = 1.458e-6*(T_0**1.5)/(T_0 + 110.4)
        c_sutherland = 120.0    # Sutherland's constant [K]
        lam = mu_0*(T_0 + c_sutherland)/(T_0**1.5)
        self.viscosity.sutherland = c_sutherland # save sutherland's constant

        # Air
        R = 287.04      # Specific gas constant [m^2/K/s**2.0]
        gamma = 1.4     # Ratio of specific heats
        # save R and gamma
        self.R  = R
        self.gamma = gamma

        # Speed of sound
        a_0 = np.sqrt(gamma*R*T_0)

        # Atmospheric calculations
        if altitude == 0.0:   # Sea level

            self.T = T_0
            self.P = P_0
            self.rho = rho_0
            self.viscosity.mu = mu_0
            self.viscosity.nu = self.viscosity.mu/self.rho
            self.a = a_0

        elif altitude <= 11000.0:    # Troposphere

            L = 0.0065

            self.T = T_0 - L*altitude
            self.P = P_0*((1.0 - L*altitude/T_0)**(g_0/(R*L)))
            self.rho = self.P/(R*self.T)
            self.viscosity.mu = (lam * self.T**1.5)/(self.T + c_sutherland)
            self.viscosity.nu = self.viscosity.mu/self.rho
            self.a = np.sqrt(gamma*R*self.T)

        elif altitude <= 20000:       # Lower Stratosphere

            T_11k = 216.65
            P_11k = 22632

            self.T = T_11k
            self.P = P_11k*np.exp(-g_0/(R*T_11k) * (altitude - 11e3))
            self.rho = self.P/(R*self.T)
            self.viscosity.mu = (lam * self.T**1.5)/(self.T + c_sutherland)
            self.viscosity.nu = self.viscosity.mu/self.rho
            self.a = np.sqrt(gamma*R*self.T)

        elif altitude <= 25000:       # Stratosphere

            self.T = 221.552
            self.P = 2549.20
            self.rho = self.P / (R * self.T)
            self.viscosity.mu = 1.448424e-5
            self.viscosity.nu = self.viscosity.mu / self.rho
            self.a = np.sqrt(gamma*R*self.T)
            
    def custom(self, g=3.721, rho=0.02, T=240, P=610, R=188.92, gamma=1.3):
        # Constants
        self.g  = g # acceleration due to gravity [m/s**2]
        self.rho= rho
        self.T  = T
        self.P  = P
        self.R  = R
        self.gamma = gamma
        self.viscosity = Viscosity()
        
        # dynamic viscosity from Sutherland's law
        if self.gas.lower() == 'air':
            mu_0    = 1.716e-5
            T_0     = 273
            S_mu    = 111
            c_sutherland = 120.
            acentric = 0.035
#            crit_T  = 131.0
#            crit_p  = 3588550.0
        else: #if self.gas.lower() in ['co2', 'carbondioxide', 'carbon dioxide']:
            mu_0    = 1.370e-5
            T_0     = 273
            S_mu    = 222
            c_sutherland = 240.
            acentric= 0.228
#            crit_T  = 304.25

        self.a  = np.sqrt(gamma*R*self.T)  
        self.viscosity.mu_0 = mu_0 # mu_ref for sutherland calc
        self.viscosity.mu_T = T_0      
        self.viscosity.mu = mu_0 * (self.T/T_0)**(3/2) * (T_0+S_mu) / (T+S_mu)
        self.viscosity.nu = self.viscosity.mu / self.rho
        self.viscosity.sutherland = c_sutherland
        self.acentric = acentric
#        self.crit_T = crit_T
#        self.crit_p = crit_p


if __name__ == '__main__':
    # Class instantiation
    environment = Environment()

    # Altitude
    test_altitude = 0.0

    # Compute environment parameters
    environment.isa(test_altitude)

    # Print output
    print('environment.T:', environment.T)
    print('environment.P:', environment.P)
    print('environment.rho:', environment.rho)
    print('sigma:', environment.rho/1.2256)
    print('environment.viscosity.mu:', environment.viscosity.mu)
    print('environment.viscosity.mu:', environment.viscosity.mu)
    print('environment.a:', environment.a)
    
    # test equations - WORKING AS EXPECTED
#    fc  = FlightCondition()
#    fc.set(test_altitude, 0, 6.5e6, 0.79, 0)
#    print('init\nu:',fc.u,'  T:',fc.ambient.T,'  m:',fc.mach,'  re:',fc.reynolds)
#    fc.MRe_to_velT()
#    print('v->m\n:',fc.u,'  T:',fc.ambient.T,'  m:',fc.mach,'  re:',fc.reynolds)
#    fc.velT_to_MRe()
#    print('m.>v\n:',fc.u,'  T:',fc.ambient.T,'  m:',fc.mach,'  re:',fc.reynolds)

#    fc.set(test_altitude, 0, 6.5e6, 0.79, 0)
#    fc.u = 150
#    print('init\nu:',fc.u,'  T:',fc.ambient.T,'  m:',fc.mach,'  re:',fc.reynolds)
#    fc.velT_to_MRe()
#    print('m.>v\n:',fc.u,'  T:',fc.ambient.T,'  m:',fc.mach,'  re:',fc.reynolds)
#    fc.MRe_to_velT()
#    print('v->m\n:',fc.u,'  T:',fc.ambient.T,'  m:',fc.mach,'  re:',fc.reynolds)

    

    # Test for environment.custom() function and mars atmos
    print('!!\tCustom environment setting test')
    environment.gas = 'co2'
    environment.custom()
    print('environment.T:', environment.T)
    print('environment.P:', environment.P)
    print('environment.rho:', environment.rho)
    print('sigma:', environment.rho/1.2256)
    print('environment.viscosity.mu:', environment.viscosity.mu)
    print('environment.viscosity.mu:', environment.viscosity.mu)
    print('environment.a:', environment.a)
    
    flightcondition = FlightCondition()
    flightcondition.atmosphere = 'mars'
    flightcondition.set(None, None, 1.1e4, 0.61, 7)
    flightcondition.ambient.custom()


