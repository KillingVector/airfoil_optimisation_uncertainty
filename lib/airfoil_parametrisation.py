import numpy as np
import bezier
import splipy as sp
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz
from scipy import special
import copy
from shapely.geometry import Polygon
from lib.contour_analysis import ContourAnalysis
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os
from csaps import csaps
from csaps import CubicSmoothingSpline

class AirfoilGeometry():

    """Summary: creates airfoil geometries and has functions to refine coordinates, calculate thickness and camber.
    This is the masterclass for the various parametrisation techniques

    Attributes:
        x (array): chordwise location of the airfoil coordinate points. sorted in xfoil style from trailing edge over
                    the top surface to the leading edge and back to trailing edge via the bottom surface
        z (array): top and bottom surface z-locations of the airfoil coordinate points (at x-locations)
        delta_z_te (float): trailing edge thickness
        thickness (array): thickness as the x_thickness location
        x_thickness (array): locations for the thickness and camber
        camber (array): camber of the airfoil at the x_thickness location
        curvature (array): curvature of the airfoil surface at the x-locations in self.x
        radius of curvature (array): inverse of curvature (derivative) at the x-locations in self.x
    """
    def __init__(self):
        # Geometry variables
        self.x = []
        self.z = []

        # Trailing edge thickness
        self.delta_z_te = 0.0

        # thickness and camber
        self.thickness = None
        self.max_thickness = 0.0
        self.min_thickness = 0.0
        self.camber = []
        self.max_camber = 0.0
        self.x_thickness = None
        self.x_max_camber = 0.0
        self.x_max_thickness = 0.0
        self.trailing_edge_angle = None

        # curvature
        self.arclength = None
        self.curvature = []
        self.x_curvature = None
        self.radius_of_curvature = []
        self.leading_edge_radius = None
        self.gradient = None

        self.leading_edge_coordinates = [0.0, 0.0]
        self.curvature_variation = []

        # centroid
        self.x_centroid = 0.0
        self.z_centroid = 0.0

        # area
        self.area = 0.0

        # inertias
        self.Ixx = None
        self.Izz = None
        self.Ixz = None

        # number of reversals
        self.nr_bottom_reversals = None
        self.nr_top_reversals = None

    def read_coordinates(self, filename, header_lines=1):
        """
        reads the coordinates from a file. While this will work with any file with two columns of text, the rest of the
        framework expects Selig-style files that start at the trailing edge, go to the leading edge over the top surface
        and then go back to the trailing edge over the bottom surface
        :param filename: name of the file to read
        :param header_lines: number of lines to skip
        :return: returns x and z coordinates of the airfoil
        """
        coordinates = np.loadtxt(filename, skiprows=header_lines)
        self.x = np.asarray(coordinates[:, 0])
        self.z = np.asarray(coordinates[:, 1])

    def interpolate_coordinates(self, nr_pts=301, order=3, resolution= 1000, D=20):
        """
        Interpolate N points whose concentration is based on curvature.
        :param nr_pts:
        :param order:
        :param N:
        :param D:
        :return:
        """
        Q = [self.x, self.z]
        res, fp, ier, msg = splprep(Q, u=None, k=order, s=1e-6, per=0, full_output=1)
        tck, u = res
        uu = np.linspace(u.min(), u.max(), resolution)
        x, z = splev(uu, tck, der=0)
        dx, dz = splev(uu, tck, der=1)
        ddx, ddz = splev(uu, tck, der=2)
        cv = np.abs(ddx * dz - dx * ddz) / (dx * dx + dz * dz) ** 1.5 + D
        cv_int = cumtrapz(cv, uu, initial=0)
        fcv = interp1d(cv_int, uu)
        cv_int_samples = np.linspace(0, cv_int.max(), nr_pts)
        u_new = fcv(cv_int_samples)
        x_new, z_new = splev(u_new, tck, der=0)

        self.x = x_new
        self.z = z_new

    def refine_coordinates(self, nr_pts=301,spline_order=3):
        """
        add points to the airfoil if the airfoil is too coarse
        uses the csapi package to fit a spline
        :param nr_pts: number of points added to the airfoil
        :param tol: tolerance parameter for the spline fitting (see ContourAnalysis)
        :return:
        """
        # make sure the number of points is odd (to avoid issues in thickness calculation)
        nr_pts = int(2*np.floor(nr_pts/2)+1)

        # rescale airfoil to go from 0 to 1
        x_min = np.min(self.x)
        x_max = np.max(self.x)
        scale_factor = x_max - x_min
        self.x -= x_min
        self.x /= scale_factor
        self.z /= scale_factor

        self.interpolate_coordinates(nr_pts=nr_pts, order=spline_order)

        # # calculate the arclength as the "u" parameter for the spline
        # # use arclength instead
        # ds = np.zeros(len(self.x))
        # for cntr in range(len(self.x) - 1):
        #     dx = self.x[cntr + 1] - self.x[cntr]
        #     dz = self.z[cntr + 1] - self.z[cntr]
        #     ds[cntr + 1] = np.sqrt(dx ** 2 + dz ** 2)
        #
        # arclength = np.cumsum(ds)
        # dsint = np.linspace(0, arclength[-1], nr_pts)
        # data = [self.x, self.z]
        # # fit the spline
        # data_i, smooth = csaps(arclength, data, dsint)
        # xi = data_i[0, :]
        # zi = data_i[1, :]
        #
        # self.x, self.z = xi, zi

    def refine_coordinates_no_scaling(self, nr_pts=301, spline_order=3):
        """
        same as refine coordinates function but does not scale the airfoil
        :param nr_pts: number of points added to the airfoil
        :param tol: tolerance parameter for the spline fitting (see ContourAnalysis)
        :return:
        """
        # make sure the number of points is odd (to avoid issues in thickness calculation)
        nr_pts = int(2 * np.floor(nr_pts / 2) + 1)

        # # calculate the arclength as the "u" parameter for the spline
        # # use arclength instead
        # ds = np.zeros(len(self.x))
        # for cntr in range(len(self.x) - 1):
        #     dx = self.x[cntr + 1] - self.x[cntr]
        #     dz = self.z[cntr + 1] - self.z[cntr]
        #     ds[cntr + 1] = np.sqrt(dx ** 2 + dz ** 2)
        #
        # arclength = np.cumsum(ds)
        # dsint = np.linspace(0, arclength[-1], nr_pts)
        # data = [self.x, self.z]
        # # fit the spline
        # data_i, smooth = csaps(arclength, data, dsint)
        # xi = data_i[0, :]
        # zi = data_i[1, :]
        # self.arclength = arclength
        #
        # self.x, self.z = xi, zi

        self.interpolate_coordinates(nr_pts=nr_pts, order=spline_order)

    def write_coordinates_to_file(self, filename):
        """
        writes coordinates into a dat file. Uses Selig format
        (trailing edge to leading edge over top surface and back to trailing edge over the bottom)
        :param filename: name of the file where coordinates will be written
        :return:
        """
        with open(filename, 'w') as f:
            f.write(os.path.split(filename)[1][:-4] + '\n')
            for i in range(len(self.x)-1):
                f.write('%s %s\n' % (self.x[i], self.z[i]))
            f.write('%s %s' % (self.x[-1], self.z[-1]))
        f.close()

    def calculate_curvature(self, nr_pts=301):
        """
        calculates the curvature of the airfoil. Based on the spline fitting algorithm in ContourAnalysis (from PyAero)
        :param tol: fitting tolerance for the spline
        :param nr_pts: number of points to calculate the curvature at
        :return:
        """
        # to avoid the RuntimeWarning
        np.seterr(divide='ignore', invalid='ignore')

        # refine the coordinates as a high number of points is needed to get a good estimate of leading edge radius

        ds = np.zeros(len(self.x))
        for cntr in range(len(self.x) - 1):
            dx = self.x[cntr + 1] - self.x[cntr]
            dz = self.z[cntr + 1] - self.z[cntr]
            ds[cntr + 1] = np.sqrt(dx ** 2 + dz ** 2)

        arclength = np.cumsum(ds)
        dsint = np.linspace(0, arclength[-1], nr_pts)

        data = [self.x, self.z]
        # set up smoothing spline
        s = CubicSmoothingSpline(arclength, data, smooth=1).spline
        ds1 = s.derivative(nu=1)
        ds2 = s.derivative(nu=2)
        ds3 = s.derivative(nu=3)

        xd = ds1(dsint)[0]
        yd = ds1(dsint)[1]
        x2d = ds2(dsint)[0]
        y2d = ds2(dsint)[1]

        x3d = ds3(dsint)[0]
        y3d = ds3(dsint)[1]

        n = xd ** 2 + yd ** 2
        d = xd * y2d - yd * x2d

        # gradient dy/dx = dy/du / dx/du
        gradient = ds1(dsint)[1] / ds1(dsint)[0]
        gradient2 = ds2(dsint)[1] / ds2(dsint)[0]
        gradient3 = ds3(dsint)[1] / ds3(dsint)[0]

        # radius of curvature
        R = n ** (3. / 2.) / abs(d)
        # curvature
        C = d / n ** (3. / 2.)

        # variation of curvature
        n2 = gradient2*n - 3*yd*y2d
        C_var = n2 / n**(5./2.)

        # test with formula from Wang
        n_w = y2d
        d_w = 1 + yd**2.
        d2_w = y3d * d_w - 3 * yd * y2d**2
        C_w = n_w / d_w**(3./2.)
        C_var_w = d2_w / d_w**(5./2.)

        if self.arclength is None:
            self.arclength = arclength

        self.curvature = C
        self.x_curvature = s(dsint)[0]
        self.gradient = gradient3
        self.radius_of_curvature = R
        self.curvature_variation = C_var
        self.leading_edge_radius = np.min(self.radius_of_curvature)

    def calc_trailing_edge_angle(self):
        # top surface
        dx_top = self.x[1] - self.x[0]
        dz_top = self.z[1] - self.z[0]
        angle_top = np.arctan(dz_top/dx_top)

        # bottom surface
        dx_bot = self.x[-2] - self.x[-1]
        dz_bot = self.z[-2] - self.z[-1]
        angle_bot = np.arctan(dz_bot / dx_bot)

        self.trailing_edge_angle = (angle_bot - angle_top)/np.pi*180


    def calc_thickness_and_camber(self, order=3,resolution=10000):
        """
        Calculate the thickness and camber of the airfoil
        :return:
        """
        # TODO use the xfoil approach to find the opposite point using the spline.
        #  Then use that to calculate thickness and camber

        Q = [self.x, self.z]
        res, fp, ier, msg = splprep(Q, u=None, k=order, s=1e-6, per=0, full_output=1)
        tck, u = res
        uu = np.linspace(u.min(), u.max(), resolution)
        x, z = splev(uu, tck, der=0)

        le_idx = np.argmin(x)
        x_le = x[le_idx]

        x_top = x[0:le_idx]
        x_bot = x[le_idx:-1]
        z_top = z[0:le_idx]
        z_bot = z[le_idx:-1]

        thickness = np.zeros(x_top.shape)

        for cntr in range(len(x_top)):
            x_to_find = x_top[cntr]
            idx = (np.abs(x_bot - x_to_find)).argmin()
            thickness[cntr] = z_top[cntr] -z_bot[idx]
        camber = z_top - thickness/2

        self.thickness = thickness[::-1]
        self.x_thickness = x_top[::-1]
        self.camber = camber[::-1]

        self.max_thickness = np.max(thickness)
        self.min_thickness = np.min(thickness)
        self.max_camber = np.max(camber)
        # get the chordwise location of maximum thickness and maximum camber points
        self.x_max_thickness = self.x_thickness[np.argmax(self.thickness)]
        self.x_max_camber = self.x_thickness[np.argmax(self.camber)]


    def local_thickness(self, x_over_c=np.linspace(0, 1, 151)):
        """
        Calculate the thickness at a set of chordwise locations
        :param x_over_c: chordwise locations where thickness needs to be known
        :return:
        """
        if self.x_thickness is None:
            self.calc_thickness_and_camber()
        return np.interp(x_over_c, self.x_thickness, self.thickness)

    def local_camber(self, x_over_c=np.linspace(0, 1, 151)):
        """
         Calculate the camber at a set of chordwise locations
        :param x_over_c: chordwise locations where thickness needs to be known
        :return:
        """
        if self.x_thickness is None:
            self.calc_thickness_and_camber()
        return np.interp(x_over_c, self.x_thickness, self.camber)

    def local_curvature(self, x_over_c=np.linspace(0, 1, 151)):
        """
         Calculate the curvature at a set of chordwise locations
        :param x_over_c: chordwise locations where thickness needs to be known
        :return:
        """
        # if not hasattr(self, 'curvature'):
        self.calculate_curvature()
        # split in top and bottom
        le_idx = np.argmin(self.x_curvature)
        top_x = self.x_curvature[0:le_idx+1]
        bot_x = self.x_curvature[le_idx:]
        top_curv = self.curvature[0:le_idx+1]
        bot_curv = self.curvature[le_idx:]
        curvature_top = np.interp(x_over_c, top_x[::-1], top_curv[::-1])
        curvature_bot = np.interp(x_over_c, bot_x, bot_curv)
        return curvature_bot, curvature_top

    def locate_leading_edge(self):
        """leading is is located as point of minimum curvature"""

        self.calculate_curvature()
        le_idx = np.argmin(self.x_curvature)
        self.calc_thickness_and_camber()
        le_thickness = self.local_thickness(self.x_curvature[le_idx])
        le_camber = self.local_camber(self.x_curvature[le_idx])
        le_z = le_camber + le_thickness / 2
        self.leading_edge_coordinates = [self.x_curvature[le_idx],le_z]


    def local_curvature_variation(self, x_over_c=np.linspace(0, 1, 151)):
        """
         Calculate the variation of curvature at a set of chordwise locations
        :param x_over_c: chordwise locations where thickness needs to be known
        :return:
        """
        if not hasattr(self, 'curvature'):
            self.calculate_curvature()
        # split in top and bottom
        le_idx = np.argmin(self.x_curvature)
        top_x = self.x_curvature[0:le_idx+1]
        bot_x = self.x_curvature[le_idx:]
        top_curv_sm = self.curvature_variation[0:le_idx+1]
        bot_curv_sm = self.curvature_variation[le_idx:]
        curvature_variation_top = np.interp(x_over_c, top_x[::-1], top_curv_sm[::-1])
        curvature_variation_bot = np.interp(x_over_c, bot_x, bot_curv_sm)
        return curvature_variation_bot, curvature_variation_top

    def calc_area(self):
        """
        Calculate the area of the airfoil
        :return:
        """
        # make a polygon
        poly = Polygon(zip(self.x, self.z))
        self.area = poly.area

    def calc_centroid(self):
        """
        Calculate the centroid of the airfoil
        :return:
        """
        poly = Polygon(zip(self.x, self.z))
        self.x_centroid = poly.centroid.x
        self.z_centroid = poly.centroid.y

    def calc_number_of_reversals(self, nr_pts=301):
        theta = np.linspace(0, 2 * np.pi, nr_pts)
        x = 0.5 * (np.cos(theta) + 1.0)
        x_vec = x[0:int(np.floor(nr_pts / 2)) + 1]

        curv_bot, curv_top = self.local_curvature(x_over_c=x_vec)

        testc = np.where(np.diff(np.sign(curv_bot) >= 0))[0]
        self.nr_bottom_reversals = int(testc.shape[0])
        testc = np.where(np.diff(np.sign(curv_top) >= 0))[0]
        self.nr_top_reversals = int(testc.shape[0])

        # # use self.gradient instead - but need to split in top and bottom
        # ind_le = np.argmin(self.x_curvature)
        # x_top = self.x_curvature[0:ind_le]
        # x_bot = self.x_curvature[ind_le:-1]
        # grad_top = self.gradient[0:ind_le]
        # grad_bot = self.gradient[ind_le:-1]

        # testc = np.where(np.diff(np.sign(grad_bot) >= 0))[0]
        # self.nr_bottom_reversals = int(testc.shape[0]/2)
        # testc = np.where(np.diff(np.sign(grad_top) >= 0))[0]
        # self.nr_top_reversals = int(testc.shape[0]/2)

    def inertia(self):
        """Moments and product of inertia about centroid."""
        # TODO need to actually run this and check if this is working properly - compare with xfoil calculations
        pts = [[self.x[i], self.z[i]] for i in range(len(self.x))]
        if pts[0] != pts[-1]:
            pts = pts + pts[:1]
        x = [c[0] for c in pts]
        y = [c[1] for c in pts]
        sxx = syy = sxy = 0
        self.calc_area()
        self.calc_centroid()
        a = self.area
        cx, cy = self.x_centroid, self.z_centroid
        for i in range(len(pts) - 1):
            sxx += (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2) * (x[i] * y[i + 1] - x[i + 1] * y[i])
            syy += (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2) * (x[i] * y[i + 1] - x[i + 1] * y[i])
            sxy += (x[i] * y[i + 1] + 2 * x[i] * y[i] + 2 * x[i + 1] * y[i + 1] + x[i + 1] * y[i]) * (
                    x[i] * y[i + 1] - x[i + 1] * y[i])
        self.Ixx = sxx / 12 - a * cy ** 2
        self.Izz = syy / 12 - a * cx ** 2
        self.Ixz = sxy / 24 - a * cx * cy

    def scale_section(self, scale_factor):
        self.x *= scale_factor
        self.z *= scale_factor

    def generate_section(self, coefficients, npt=301, delta_z_te=0.0, **kwargs):
        """
        Generate an airfoil section (used in airfoil_fitting)
        :param coefficients: shape parameters of the parametrisation scheme that is used
        :param npt: number of points on the airfoil
        :param delta_z_te: trailing edge thickness
        :param kwargs: additional parameters as required by the specific parametrisation scheme
        :return:
        """
        raise NotImplementedError('the parametrisation class needs to have this implemented')


class CST(AirfoilGeometry):
    """
    Class-Shape Transformation airfoil parametrisation scheme by Kulfan of Boeing. See google for an
    explanation of the actual scheme
    """
    def __init__(self):
        super().__init__()

        # Weighting of Bernstein binomials
        self.b_upper = []
        self.b_lower = []

        # Order of Bernstein binomials
        self.order = 0

    def generate_section(self, coefficients, npt=201, delta_z_te=0.0, **kwargs):

        b_upper = coefficients[0:int(np.floor(len(coefficients) / 2))]
        b_lower = coefficients[int(np.floor(len(coefficients) / 2)):]

        # Setting geometric variables
        self.x, x_upper, x_lower = generate_cosine_spacing(npt)
        z_upper = np.zeros(np.shape(x_upper))
        z_lower = np.zeros(np.shape(x_lower))

        # Assigning weighting of Bernstein binomials
        self.b_upper = b_upper
        self.b_lower = b_lower
        self.delta_z_te = delta_z_te

        # Calculating order of Bernstein binomials
        self.order = np.shape(b_upper)[0] - 1

        # Class parameters (n_1 = 0.5, n_2 = 1.0 for airfoils)
        n_1 = 0.5
        n_2 = 1.0

        # Computing class functions
        c_upper = (x_upper ** n_1) * ((1.0 - x_upper) ** n_2)
        c_lower = (x_lower ** n_1) * ((1.0 - x_lower) ** n_2)

        # Computing k terms
        k = np.zeros(self.order + 1)
        for i in range(self.order + 1):
            k[i] = np.math.factorial(self.order) / (np.math.factorial(i) * np.math.factorial(self.order - i))

        # Computing shape functions
        s_upper = np.zeros(np.shape(x_upper))
        s_lower = np.zeros(np.shape(x_lower))
        for i in range(np.shape(x_upper)[0]):
            for j in range(np.shape(self.b_upper)[0]):
                s_upper[i] += self.b_upper[j] * k[j] * (x_upper[i] ** j) * (
                        (1.0 - x_upper[i]) ** (self.order - j))
        for i in range(np.shape(x_lower)[0]):
            for j in range(np.shape(self.b_lower)[0]):
                s_lower[i] += self.b_lower[j] * k[j] * (x_lower[i] ** j) * (
                        (1.0 - x_lower[i]) ** (self.order - j))

        # Computing airfoil z coordinates
        for i in range(np.shape(x_upper)[0]):
            z_upper[i] = c_upper[i] * s_upper[i] + x_upper[i] * self.delta_z_te/2
        for i in range(np.shape(x_lower)[0]):
            z_lower[i] = c_lower[i] * s_lower[i] - x_lower[i] * self.delta_z_te/2

        # Form the upper and lower airfoil surfaces
        self.z = np.concatenate((z_upper, z_lower[1:]))


class CSTmod(AirfoilGeometry):
    """
    Modified CST scheme to allow a better representation of the leading edge of the airfoil
    See: D. Masters, D. Poole, N. Tayloar, T. Rendall, C. Allen, Influence of shape parameterization on a benchmark
    aerodynamic optimization problem, Journal of Aircraft 54 (6) (2017) 2242–2256.
    """
    def __init__(self):
        super().__init__()

        # Leading edge parameter
        self.b_leading_edge = []

        # Weighting of Bernstein binomials
        self.b_upper = []
        self.b_lower = []

        # Order of Bernstein binomials
        self.order = 0

    def generate_section(self, coefficients, npt=201, delta_z_te=0.0, x_vec=None, **kwargs):

        if (len(coefficients) % 2) == 1:
            self.order = int((len(coefficients) - 3) / 2)
            self.b_upper = coefficients[0:self.order + 1]
            self.b_lower = coefficients[self.order + 1:-1]
            self.b_leading_edge = coefficients[-1]
            self.delta_z_te = delta_z_te
        else:
            self.order = int((len(coefficients) - 4) / 2)
            self.b_upper = coefficients[0:self.order + 1]
            self.b_lower = coefficients[self.order + 1:-2]
            self.b_leading_edge = coefficients[-2]
            self.delta_z_te = coefficients[-1]

        # Setting geometric variables
        if x_vec is None:
            self.x, x_upper, x_lower = generate_cosine_spacing(npt)
        else:
            ind_le = np.argmin(x_vec)
            self.x = x_vec
            x_upper = x_vec[0:ind_le+1]
            x_lower = x_vec[ind_le:]
        z_upper = np.zeros(np.shape(x_upper))
        z_lower = np.zeros(np.shape(x_lower))

        # Class parameters (n_1 = 0.5, n_2 = 1.0 for airfoils)
        n_1 = 0.5
        n_2 = 1.0

        # Computing class functions
        c_upper = (x_upper ** n_1) * ((1.0 - x_upper) ** n_2)
        c_lower = (x_lower ** n_1) * ((1.0 - x_lower) ** n_2)

        # Computing k terms
        k = np.zeros(self.order + 1)
        for i in range(self.order + 1):
            k[i] = np.math.factorial(self.order) / (np.math.factorial(i) * np.math.factorial(self.order - i))

        # Computing shape functions
        s_upper = np.zeros(np.shape(x_upper))
        s_lower = np.zeros(np.shape(x_lower))
        for i in range(np.shape(x_upper)[0]):
            for j in range(np.shape(self.b_upper)[0]):
                s_upper[i] += self.b_upper[j] * k[j] * (x_upper[i] ** j) * (
                        (1.0 - x_upper[i]) ** (self.order - j))
        for i in range(np.shape(x_lower)[0]):
            for j in range(np.shape(self.b_lower)[0]):
                s_lower[i] += self.b_lower[j] * k[j] * (x_lower[i] ** j) * (
                        (1.0 - x_lower[i]) ** (self.order - j))

        # Computing airfoil z coordinates
        for i in range(np.shape(x_upper)[0]):
            z_upper[i] = c_upper[i] * s_upper[i] + x_upper[i] * self.delta_z_te/2
            z_upper[i] = z_upper[i] + self.b_leading_edge * (x_upper[i] ** 0.5) * (
                    (1 - x_upper[i]) ** (self.order - 0.5))
        for i in range(np.shape(x_lower)[0]):
            z_lower[i] = c_lower[i] * s_lower[i] - x_lower[i] * self.delta_z_te/2
            z_lower[i] = z_lower[i] + self.b_leading_edge * (x_lower[i] ** 0.5) * (
                    (1 - x_lower[i]) ** (self.order - 0.5))

        # Form the upper and lower airfoil surfaces
        self.z = np.concatenate((z_upper, z_lower[1:]))


class BezierAirfoil(AirfoilGeometry):
    """
    Bezier airfoil parametrisation class.
    Bezier splines give more global control than B-splines and are thus less sensitive to local oscillations
    """
    def __init__(self):
        super().__init__()

        # y- location of the control points
        self.b_upper = []
        self.b_lower = []

    def generate_section(self, coefficients, npt=201, delta_z_te=0.0, x_vec=None, **kwargs):

        b_upper = coefficients[0:int(np.floor(len(coefficients) / 2))]
        b_lower = coefficients[int(np.floor(len(coefficients) / 2)):]

        # check if coefficients are given as array or not
        if not isinstance(b_upper, np.ndarray):
            b_upper = np.asfortranarray(b_upper)
        if not isinstance(b_lower, np.ndarray):
            b_lower = np.asfortranarray(b_lower)

        # Setting geometric variables
        if x_vec is None:
            s, s_upper, s_lower = generate_cosine_spacing(npt)

        else:
            ind_le = np.argmin(x_vec)
            s = x_vec
            s_upper = x_vec[0:ind_le+1]
            s_lower = x_vec[ind_le:]

        # Setting up control point chord-wise location
        nr_ctrl_points = len(b_lower) +1
        x_ctrl = generate_control_point_spacing(nr_ctrl_points)

        # Assigning y location of the control points
        self.b_upper = b_upper
        self.b_lower = b_lower
        self.delta_z_te = delta_z_te

        # add control point for leading edge
        x_ctrl = np.insert(x_ctrl, 0, 0.0)
        b_upper = np.insert(b_upper, 0, 0.0)
        b_lower = np.insert(b_lower, 0, 0.0)

        # add control point for trailing edge - no need for x as that is already done by raising nr_ctrl_points by one
        b_upper = np.append(b_upper, self.delta_z_te / 2)
        b_lower = np.append(b_lower, -self.delta_z_te / 2)

        # Generating the control points
        nodes_upper = np.asfortranarray([x_ctrl, b_upper])
        nodes_lower = np.asfortranarray([x_ctrl, b_lower])

        # Generating the Bezier curves
        curve_upper = bezier.Curve.from_nodes(nodes_upper)
        curve_lower = bezier.Curve.from_nodes(nodes_lower)

        # Computing airfoil z and x coordinates
        z_upper = curve_upper.evaluate_multi(s_upper)[:][1]
        z_lower = curve_lower.evaluate_multi(s_lower)[:][1]

        x_upper = curve_upper.evaluate_multi(s_upper)[:][0]
        x_lower = curve_lower.evaluate_multi(s_lower)[:][0]

        # combine the surfaces
        self.z = np.concatenate((z_upper, z_lower[1:]))
        self.x = np.concatenate((x_upper, x_lower[1:]))


class BsplineAirfoil(AirfoilGeometry):
    """
    Bspline airfoil parametrisation scheme. Can be sensitive to local oscillations so might need constraints on
    curvature and variation of curvature when used in an optimisation
    """
    def __init__(self):
        super().__init__()

        # z-location of the control points
        self.b_upper = []
        self.b_lower = []

        # Order of b-spline curve
        self.order = 3

    def generate_section(self, coefficients, npt=201, delta_z_te=0.0, order=3, **kwargs):

        b_upper = coefficients[0:int(np.floor(len(coefficients) / 2))]
        b_lower = coefficients[int(np.floor(len(coefficients) / 2)):]

        # check if coefficients are given as array or not
        if not isinstance(b_upper, np.ndarray):
            b_upper = np.asfortranarray(b_upper)
        if not isinstance(b_lower, np.ndarray):
            b_lower = np.asfortranarray(b_lower)

        # Setting geometric variables
        s, s_upper, s_lower = generate_cosine_spacing(npt)

        # Setting up control point chord-wise location
        nr_ctrl_points = len(b_lower) + 1
        x_ctrl = generate_control_point_spacing(nr_ctrl_points)

        # Assigning y location of the control points
        self.b_upper = b_upper
        self.b_lower = b_lower
        self.delta_z_te = delta_z_te
        self.order = order


        # add control point for leading edge
        x_ctrl = np.insert(x_ctrl, 0, 0.0)
        b_upper = np.insert(b_upper, 0, 0.0)
        b_lower = np.insert(b_lower, 0, 0.0)

        # add control point for trailing edge - no need for x as that is already done by raising nr_ctrl_points by one
        b_upper = np.append(b_upper, self.delta_z_te / 2)
        b_lower = np.append(b_lower, self.delta_z_te / 2)

        # create knots for upper and lower surface
        knots_sequence = np.linspace(0.0, 1.0, nr_ctrl_points - (self.order + 1) + 4)
        knots_upper = np.zeros(self.order - 1)
        for cntr in range(len(knots_sequence)):
            knots_upper = np.append(knots_upper, knots_sequence[cntr])
        knots_upper = np.append(knots_upper, np.ones(self.order - 1))
        knots_lower = knots_upper

        # create basis for upper and lower surface
        basis_upper = sp.BSplineBasis(order=self.order, knots=knots_upper)
        basis_lower = sp.BSplineBasis(order=self.order, knots=knots_lower)

        # create the list of control points
        control_points_upper = []
        for cntr in range(len(b_upper)):
            control_points_upper.append([x_ctrl[cntr], b_upper[cntr]])

        control_points_lower = []
        for cntr in range(len(b_lower)):
            control_points_lower.append([x_ctrl[cntr], b_lower[cntr]])

        # make the curves
        curve_upper = sp.Curve(basis_upper, control_points_upper)
        curve_lower = sp.Curve(basis_lower, control_points_lower)

        # evaluate the curves
        coords_upper = curve_upper.evaluate(s_upper)
        coords_lower = curve_lower.evaluate(s_lower)

        # Computing airfoil z and x coordinates
        z_upper = coords_upper[:, 1]
        z_lower = coords_lower[:, 1]

        x_upper = coords_upper[:, 0]
        x_lower = coords_lower[:, 0]

        # combine the surfaces
        self.z = np.concatenate((z_upper, z_lower[1:]))
        self.x = np.concatenate((x_upper, x_lower[1:]))


class HicksHenneAirfoil(AirfoilGeometry):
    """
    Airfoil parametrisation using Hicks-Henne bump functions. See literature for a description
    Adds bump function on top of an existing airfoil. NACA 4 series airfoil is used as default but can be modified by
    passing an airfoil as a keyword argument in the generate_section function
    """
    def __init__(self):
        super().__init__()

        # y- location of the control points
        self.b_upper = []
        self.b_lower = []

        # number of bumps
        self.nr_bumps = []

        # base airfoil
        self.base_airfoil = []

    def generate_section(self, coefficients, npt=201, delta_z_te=0.0, base_airfoil='naca2412', **kwargs):

        b_upper = coefficients[0:int(np.floor(len(coefficients) / 2))]
        b_lower = coefficients[int(np.floor(len(coefficients) / 2)):]

        # check if coefficients are given as array or not
        if not isinstance(b_upper, np.ndarray):
            b_upper = np.asfortranarray(b_upper)
        if not isinstance(b_lower, np.ndarray):
            b_lower = np.asfortranarray(b_lower)

        self.b_upper = b_upper
        self.b_lower = b_lower
        self.base_airfoil = base_airfoil

        self.nr_bumps = len(b_upper)

        # load the base airfoil and re-spline it
        base_af = AirfoilGeometry()
        match = base_airfoil.find('naca')
        # match = re.search('naca', self.base_airfoil[-12:-8])
        if match != -1:
            npts = int(np.floor(npt/2-1))
            naca_number = base_airfoil[match+4:match+8]
            base_af.x, base_af.z = create_naca_4digits(naca_number, npts)
        else:
            base_af.read_coordinates(base_airfoil)
        base_af.refine_coordinates(npt)
        x_coord = base_af.x
        y_coord = base_af.z

        # rescale coordinates of base-airfoil
        if np.min(x_coord) != 0.0:
            x_min = np.min(x_coord)
            x_max = np.max(x_coord)
            scale_factor = x_max - x_min
            x_coord -= x_min
            x_coord /= scale_factor
            y_coord /= scale_factor

        # extract coordinates for lower and upper surface
        ind_min = np.argmin(x_coord)
        x_upper = x_coord[0:ind_min + 1]
        z_upper = y_coord[0:ind_min + 1]
        x_upper = x_upper[::-1]
        z_upper = z_upper[::-1]
        x_lower = x_coord[ind_min:]
        z_lower = y_coord[ind_min:]

        # Generate the bumps for the top surface
        # bumps are half-cosine space as suggested in Masters2017
        x_top = np.arange(1, self.nr_bumps + 1, 1)
        x_mi = 0.5 * (1.0 - np.cos(x_top * np.pi / (self.nr_bumps + 1)))
        mag = b_upper
        w = 1  # width of bums is set to chord as suggested in Masters2017
        m = np.log(0.5) / np.log(x_mi)

        b_top = np.zeros(len(x_upper))
        for cntr1 in range(len(x_upper)):
            for cntr2 in range(len(m)):
                b_top[cntr1] += mag[cntr2] * np.sin(np.pi * x_upper[cntr1] ** m[cntr2]) ** w

        # Generate the bumps for the top surface
        # bumps are half-cosine space as suggested in Masters2017
        x_bot = np.arange(1, self.nr_bumps + 1, 1)
        x_mi = 0.5 * (1.0 - np.cos(x_bot * np.pi / (self.nr_bumps + 1)))
        mag = b_lower
        w = 1  # width of bums is set to chord as suggested in Masters2017
        m = np.log(0.5) / np.log(x_mi)

        b_bot = np.zeros(len(x_lower))
        for cntr1 in range(len(x_lower)):
            for cntr2 in range(len(m)):
                b_bot[cntr1] += mag[cntr2] * np.sin(np.pi * x_lower[cntr1] ** m[cntr2]) ** w

        # create coordinates
        z_upper = z_upper[::-1] + b_top
        z_lower = z_lower - b_bot

        # add delta_z_te
        z_upper = z_upper + x_upper*delta_z_te/2
        z_lower = z_lower - x_lower*delta_z_te/2

        # Concatenate
        z_coord = np.concatenate((z_upper, z_lower[1:]))

        # transfer it back into self
        self.z = z_coord
        self.x = x_coord


class ParsecAirfoil(AirfoilGeometry):
    """
    Parsec airfoil parametrisation class
    Is limited as it only takes 12 coordinates but each coordinate has a physical meaning so it is easier to interpret
    than most of the other schemes
    """
    def __init__(self):
        super().__init__()

    def generate_section(self, coefficients, npt=201, delta_z_te=0.0, **kwargs):

        # check if coefficients are given as array or not
        p = coefficients
        if not isinstance(p, np.ndarray):
            p = np.asarray(p)

        # based on Kanazaki2014
        # coefficients are:
        # p1    - leading edge radius
        # p2    - max thickness location (x_t)
        # p3    - max thickness
        # p4    - curvature at max thickness location
        # p5    - trailing edge thickness
        # p6    - trailing edge wedge angle
        # p7    - camber radius of leading edge (modified version)
        # p8    - max camber location (x_c)
        # p9    - max camber
        # p10   - curvature at max camber location
        # p11   - trailing edge slope
        # p12   - trailing edge offset

        if len(p) == 11:
            p = np.append(p, 0)

        # Defining thickness coefficient matrix
        C_t = np.array([[1, 1, 1, 1, 1, 1],
                        [p[1] ** 0.5, p[1] ** 1.5, p[1] ** 2.5, p[1] ** 3.5, p[1] ** 4.5, p[1] ** 5.5],
                        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                        [0.5 * p[1] ** (-0.5), 1.5 * p[1] ** 0.5, 2.5 * p[1] ** 1.5, 3.5 * p[1] ** 2.5,
                         4.5 * p[1] ** 3.5, 5.5 * p[1] ** 4.5],
                        [-0.25 * p[1] ** (-1.5), 0.75 * p[1] ** (-0.5), 3.75 * p[1] ** 0.5, 8.75 * p[1] ** 1.5,
                         15.75 * p[1] ** 2.5, 24.75 * p[1] ** 2.5],
                        [1, 0, 0, 0, 0, 0]])

        # Defining camber coefficient matrix
        C_c = np.array([[1, 1, 1, 1, 1, 1],
                        [p[7] ** 0.5, p[7] ** 1, p[7] ** 2, p[7] ** 3, p[7] ** 4, p[7] ** 5],
                        [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                        [0.5 * p[7] ** (-0.5), 1.0 * p[7] ** 0.0, 2.0 * p[7] ** 1.0, 3.0 * p[7] ** 2.0,
                         4.0 * p[7] ** 3.0, 5.0 * p[7] ** 4.0],
                        [-0.25 * p[7] ** (-1.5), 0, 2.0 * p[7] ** 0.0, 6.0 * p[7] ** 1.0, 12.0 * p[7] ** 2.0,
                         20.0 * p[7] ** 3.0],
                        [1, 0, 0, 0, 0, 0]])

        # Defining upper right-hand-size vector
        b_t = np.array([[0.5 * p[4], p[2], np.tan(p[5]), 0, p[3], (2 * p[0]) ** 0.5]])

        # Defining lower right-hand-size vector
        b_c = np.array([[p[11], p[8], np.tan(p[10]), 0, p[9], np.sign(p[6]) * ((2 * p[6]) ** 0.5)]])

        # Calculating upper and lower a coefficients
        a_t = np.linalg.solve(C_t, b_t.T)
        a_c = np.linalg.solve(C_c, b_c.T)

        # Defining x vectors
        theta = np.linspace(0, 2 * np.pi, npt)
        x = 0.5 * (np.cos(theta) + 1)
        x_v = x[:int(np.floor(npt / 2) + 1)]

        # Forming upper and lower surfaces of the airfoil

        camber = a_c[0] * x_v ** 0.5

        for i in range(5):
            camber += a_c[i + 1] * x_v ** (i + 1)

        thickness = np.zeros(len(x_v))
        for i in range(len(a_t)):
            thickness += a_t[i] * x_v ** ((2 * (i + 1) - 1) / 2)

        z_upper = camber + 0.5 * thickness
        z_lower = camber - 0.5 * thickness
        z_lower = z_lower[::-1]
        # Concatenate
        self.z = np.concatenate((z_upper, z_lower[1:]))
        self.x = x


def create_naca_4digits(number, n=201, finite_te=False, half_cosine_spacing=True):
    """
    Returns 2*n+1 points in [0 1] for the given 4 digit NACA number string
    """
    if len(number) != 4 or not number.isdigit():
        raise ValueError('Only 4 digit naca airfoils allowed (as string)')

    m = float(number[0]) / 100.0
    p = float(number[1]) / 10.0
    t = float(number[2:]) / 100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843

    if finite_te:
        a4 = -0.1015  # For finite thick TE
    else:
        a4 = -0.1036  # For zero thick TE

    if half_cosine_spacing:
        beta = np.linspace(0.0, np.pi, n + 1)
        x = 0.5 * (1.0 - np.cos(beta))  # Half cosine based spacing
    else:
        x = np.linspace(0.0, 1.0, n + 1)

    yt = 5 * t * (a0 * np.sqrt(x) + a1 * x + a2 * pow(x, 2) + a3 * pow(x, 3) + a4 * pow(x, 4))

    xc1 = x[x <= p]
    xc2 = x[x > p]

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = -yt
    else:
        yc1 = m / pow(p, 2) * xc1 * (2 * p - xc1)
        yc2 = m / pow(1 - p, 2) * (1 - 2 * p + xc2) * (1 - xc2)
        zc = np.concatenate((yc1, yc2))

        dyc1_dx = m / pow(p, 2) * (2 * p - 2 * xc1)
        dyc2_dx = m / pow(1 - p, 2) * (2 * p - 2 * xc2)
        dyc_dx = np.concatenate((dyc1_dx, dyc2_dx))

        theta = np.arctan(dyc_dx)

        xu = x - yt * np.sin(theta)
        yu = zc + yt * np.cos(theta)

        xl = x + yt * np.sin(theta)
        yl = zc - yt * np.cos(theta)

    X = np.concatenate((xu[::-1], xl[1:]))
    Y = np.concatenate((yu[::-1], yl[1:]))
    return X, Y


def initialise_airfoil(parametrisation):
    """
    function to create instances of the different parametrisation schemes
    :param parametrisation: parametrisation class to be used
    :return: instance of the class
    """
    if parametrisation.lower() == 'bezier':
        airfoil = BezierAirfoil()
    elif parametrisation.lower() == 'bspline':
        airfoil = BsplineAirfoil()
    elif parametrisation.lower() == 'cstmod':
        airfoil = CSTmod()
    elif parametrisation.lower() == 'hickshenne':
        airfoil = HicksHenneAirfoil()
    elif parametrisation.lower() == 'parsec':
        airfoil = ParsecAirfoil()
    elif parametrisation.lower() == 'cst':
        airfoil = CST()
    else:
        airfoil = None
    return airfoil


def generate_cosine_spacing(npt):
    """
    Generate cosine spacing along the chord
    :param npt: number of points to be generated
    :return: x: list of chordwise point locations
    :return: x_upper: list of chordwise point locations on upper surface
    :return: x_lower: list of chordwise point locations on lower surface
    """
    theta = np.linspace(0.0, 2.0 * np.pi, npt)
    x = 0.5 * (np.cos(theta) + 1.0)
    x_upper = x[0:int(np.floor(npt / 2)) + 1]
    x_lower = x[int(np.floor(npt / 2)):]
    return x, x_upper, x_lower


def generate_control_point_spacing(nr_ctrl_points):
    """
    Generates cosine spacing for the control points used in BezierAirfoil and BsplineAirfoil
    :param nr_ctrl_points: number of control points to be generated
    :return: x_ctrl: chordwise location of the control points
    """
    theta_ctrl = np.linspace(np.pi, 0.0, nr_ctrl_points)
    x_ctrl = 0.5 * (np.cos(theta_ctrl) + 1.0)
    return x_ctrl


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt


    # Plot settings
    # mpl.rc('lines', linewidth=2, markersize=10)
    # mpl.rc('axes', labelsize=17)
    # mpl.rc('xtick', labelsize=17)
    # mpl.rc('ytick', labelsize=17)
    # mpl.rc('legend', fontsize=15)
    # mpl.rc('figure', figsize=[6.4, 4.8])
    mpl.rc('savefig', dpi=300, format='pdf', bbox='tight')

    # Number of airfoil points
    nr_of_pts = 201

    # Initiating cst class instance
    section = CSTmod()
    # section = BezierAirfoil()
    # section = BsplineAirfoil()
    # section = HicksHenneAirfoil()
    # section = ParsecAirfoil()
    # Coefficient arrays
    b_upper = np.array([0.2737, 0.3459, 0.3746, 0.3594, 0.2779, 0.2761, 0.2810])
    b_lower = np.array([-0.2965, -0.1349, -0.1873, -0.1322, -0.1327, -0.1699, -0.2604])
    # b_lower = np.array([0.2965, 0.1349, 0.1873, 0.1322, 0.1327, 0.1699, 0.2604])

    airfoil_definition = [0.0837961618809277, 0.390802420521655, 0.209726369541402, -2.74967855872026, 0,
                          -0.165756238952580, 0.00128490952503425, 0.489546487934853, 0.0221595476027768,
                          -0.121888487777930, -0.141588967719469]

    # Calculating airfoil section coordinates
    test_coefficients = np.concatenate((b_upper,b_lower))


    test_coefficients = [2.012062279244094, 1.2318617501607945, 0.5986000786331532, 0.13089042306868215, 1.6843595271176417,
                    0.9487363833839282, 0.23214060801383257, 0.07942600246767, -1.7709169928531763]
    section.generate_section(test_coefficients, nr_of_pts, 0)

    # Hicks Henne
    # base_airfoil = 'baseairfoil/naca2412.dat'
    # section.generate_section(coefficients, npt, 0, base_airfoil)

    # Parsec
    # section.generate_section(airfoil_definition, npt)

    fig, ax = plt.subplots()
    ax.plot(section.x, section.z, color='C0', label='Test section')
    # ax.plot(section.x_upper, section.z_upper, color='C1', label='Test section')
    # ax.plot(section.x_lower, section.z_lower, color='C2', label='Test section')
    ax.set_xlabel('x [--]')
    ax.set_ylabel('z [--]')
    ax.grid(True)
    ax.legend()
    ax.axis('equal')
    # plt.savefig('../figures/airfoils' + '.pdf')

    plt.show()
#    plt.close()


