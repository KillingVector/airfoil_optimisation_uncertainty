General
-------
This is a general framework for the optimisation of single element airfoils. The framework allows the use of a range of 
optimisation algorithms and couples these with a variety of aerodynamic analysis tools. It is intended to be 
modular and flexible.

An interface is provided to: 
- Xfoil: Mark Drela's panel code coupled with a e^n boundary layer solver.
- SU2: Stanford University's Unstructured RANS (Reynolds-Averaged Navier-Stokes) solver. Works well at medium to high 
    Reynolds number and for compressible flow. Is much faster than OpenFOAM but only has limited transitional turbulence
    models. So should not be the first choice if the application is low Reynolds number.
- OpenFOAM: Another widely-used and very popular RANS package. Has a lot of transitional turbulence models so it the 
    number one choice for high-fidelity low Reynolds number flows.
- MSES: Another tool from Mark Drela that can solve compressible flows as well as multi-element (not available here)    

Main files to set up a new optimisation run
-------------------------------------------
The framework offers a lot of functionality that should not be touched unless you are developing parts of it or adding 
new capability. 

To set up a new case you should only make changes here:
- main.py: this is where you choose the algorithm, set the solver, choose the parametrisation scheme (and its order). 
        Set the config.case_name here so that the right case is selected. Details of the case name are set in design.py
- lib/design.py: this is the main file where you set up a new case. Plenty of example cases are provided. They are 
        selected using the config.case_name that you set in main.py. You should copy an existing case and change the 
        relevant operating conditions and constraints for your case. You need to check constraints!
- lib/solver_settings.py: this is where you set the location of the various solvers!! You do need to change these to 
        reflect your specific machine setup. If you do not make changes here the framework will not find xfoil, SU2, ...


Some additional files that you might need to look at if you want to make substantial changes are:
- lib/simulation.py: this is where you make more substantial changes to the actual running of the case. Here is where 
        you would add additional functions to calculate constraints, change the way objectives are calculated, ...
- cases/single_element_setup.py: if you want to make more significant changes to the way the optimisation is run. If you 
    want to add new types of variables or constraints, this is the place to do it. Use only if you need substantial 
    changes 

Airfoil Fitting 
---------------
As a special use case the framework can be used to smooth out existing airfoil coordinates files. 
See lib/airfoil_fitting.py for an example (at the bottom). If your airfoil coordinates are rotated (trailing edge is
not at 0 z) this will use Xfoil to de-rotate the airfoil so you should have the path for xfoil set up correctly in 
lib/solver_settings.py

Airfoil Parametrisation 
-----------------------
A brief overview of the available parametrisation schemes is given here. You should read up on the various schemes in 
literature if you are using these. Also have a look at lib/airfoil_parametrisation.py for the implementation and a bit
more information. 

- CST: class-shape transformation class of Brenda Kulfan from Boeing. Widely used and probably the first point of call
- CSTmod: modified version of CST to give more control to the leading edge
- Bezier: Bezier splines offer good global control of the shape but need larger variations in parameter values
- B-spline: B-splines offer much more local control but as a consequence can lead to oscillations 
- Parsec: popular because it uses geometric parameters rather than abstract values as parameters. Limited because it 
    only has 12 parameters, which can not be changed 
- Hicks-Henne: bump functions added to a baseline airfoil    

Optimisation Algorithms
-----------------------
A brief overview of the main algorithms is given here. You should read up on the details in literature and familiarise 
yourself with the algorithms in literature before you change from the defaults.

Available algorithms at the time of writing this readme file are:
- NSGA2: should be the default algorithm and is the standard set up in main.py. NSGA2 is a non-dominated sorting 
    genetic algorithm which selects individuals from the non-dominated front based on the crowding distance. It works 
    well, is widely used but can be slower to converge than some of the alternatives. Unless you understand the 
    subtleties of the algorithms this should be your first choice if you have two objectives
- NSGA3: is an extension of NSGA2 to deal with more than two objectives (a so-called many-objective problem). It uses 
    reference vector based selection of the non-dominated front (instead of crowding-distance based selection in NSGA2)
- CTAEA: another evolutionary algorithm with reference vectors. Makes use of an external algorithm. See the paper below
    Ke Li, Renzhi Chen, Guangtao Fu, and Xin Yao. Two-Archive Evolutionary Algorithm for Constrained Multiobjective 
    Optimization. IEEE Transactions on Evolutionary Computation, 23(2):303–315, April 2019. 
    doi:10.1109/TEVC.2018.2855411.
- MOEAD: under development
- NSMTLBO: a teaching-learning based algorithm that is currently under development
- others that are less tested and less documented

To Do's
--------
- test MSES
- add some booleans to allow an easier choice of compressible vs incompressible, turbulence models, ... 
    (add a simulation_settings.py file to select those rather than setting them in main.py? - if yes update in this 
    readme)
- update some naming of the functions for OpenFOAM and SU2 to make it compatible with PEP8
- integrate surrogate modeling framework
- update optimisation framework if extra algorithms are tested
- see various #TODO 

        