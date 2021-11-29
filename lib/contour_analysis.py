import numpy as np
import scipy.interpolate as si
import copy


class ContourAnalysis(object):
    """Summary

    Attributes:
        canvas (TYPE): Description
        curvature_data (TYPE): Description
        figure (TYPE): Description
        parent (QMainWindow object): MainWindow instance
        raw_coordinates (list): contour points as tuples
        spline_data (TYPE): Description
        toolbar (TYPE): Description
    """
    def __init__(self):

        self.spline_data = None
        self.curvature_data = None
        self.leading_edge_radius = None

    def spline(self, x, y, points=501, degree=3, evaluate=False):
        """Interpolate spline through given points

        Args:
            x, y: coordinates of the object that needs to be splined
            points (int, optional): Number of points on the spline
            degree (int, optional): Degree of the spline
            evaluate (bool, optional): If True, evaluate spline just at
                                       the coordinates of the knots
        """

        # interpolate B-spline through data points
        # returns knots of control polygon
        # tck ... tuple (t,c,k) containing the vector of knots,
        # the B-spline coefficients, and the degree of the spline.
        # u ... array of the parameters for each knot
        tck, u = si.splprep([x, y], s=0, k=degree)

        # number of points on interpolated B-spline (parameter t)
        t = np.linspace(0.0, 1.0, points)

        # if True, evaluate spline just at the coordinates of the knots
        if evaluate:
            t = u

        # evaluate B-spline at given parameters
        # der=0: returns point coordinates
        coo = si.splev(t, tck, der=0)

        # evaluate 1st derivative at given parameters
        der1 = si.splev(t, tck, der=1)

        # evaluate 2nd derivative at given parameters
        der2 = si.splev(t, tck, der=2)

        # evaluate 2nd derivative at given parameters
        der3 = si.splev(t, tck, der=3)

        self.spline_data = [coo, u, t, der1, der2, tck, der3]

        return self.spline_data

    def refine(self, tol=170.0, recursions=0, verbose=False):
        """Recursive refinement with respect to angle criterion (tol).
        If angle between two adjacent line segments is less than tol,
        a recursive refinement of the contour is performed until
        tol is met.

        Args:
            tol (float, optional): Angle between two adjacent contour segments
            recursions (int, optional): NO USER INPUT HERE
                                        Needed just for level information
                                        during recursions
            verbose (bool, optional): Activate logger messages
        """

        # self.spline_data = [coo, u, t, der1, der2, tck]
        xx, yy = self.spline_data[0]
        t = self.spline_data[2]
        tck = self.spline_data[5]

        if verbose:
            logger.log.info('\nPoints before refining: %s \n' % (len(xx)))

        xn = copy.deepcopy(xx)
        yn = copy.deepcopy(yy)
        tn = copy.deepcopy(t)

        j = 0
        refinements = 0
        first = True
        refined = dict()

        for i in range(len(xx) - 2):
            refined[i] = False

            # angle between two contour line segments
            a = np.array([xx[i], yy[i]])
            b = np.array([xx[i + 1], yy[i + 1]])
            c = np.array([xx[i + 2], yy[i + 2]])
            angle = angle_between(a - b, c - b, degree=True)

            if angle < tol:

                refined[i] = True
                refinements += 1

                # parameters for new points
                t1 = (t[i] + t[i + 1]) / 2.
                t2 = (t[i + 1] + t[i + 2]) / 2.

                # coordinates of new points
                p1 = si.splev(t1, tck, der=0)
                p2 = si.splev(t2, tck, der=0)

                # insert points and their parameters into arrays
                if i > 0 and not refined[i - 1]:
                    xn = np.insert(xn, i + 1 + j, p1[0])
                    yn = np.insert(yn, i + 1 + j, p1[1])
                    tn = np.insert(tn, i + 1 + j, t1)
                    j += 1
                xn = np.insert(xn, i + 2 + j, p2[0])
                yn = np.insert(yn, i + 2 + j, p2[1])
                tn = np.insert(tn, i + 2 + j, t2)
                j += 1

                if first and recursions > 0:
                    if verbose:
                        logger.log.info('Recursion level: %s \n' %
                                        (recursions))
                    first = False

                if verbose:
                    logger.log.info('Refining between %s %s, Tol=%05.1f Angle=%05.1f\n'
                                    % (i, i + 1, tol, angle))

        if verbose:
            logger.log.info('Points after refining: %s' % (len(xn)))

        # update coordinate array, including inserted points
        self.spline_data[0] = (xn, yn)
        # update parameter array, including parameters of inserted points
        self.spline_data[2] = tn

        # recursive refinement
        if refinements > 0:

            self.refine(tol, recursions + 1, verbose)

        # stopping from recursion if no refinements done in this recursion
        else:

            # update derivatives, including inserted points
            self.spline_data[3] = si.splev(tn, tck, der=1)
            self.spline_data[4] = si.splev(tn, tck, der=2)
            self.spline_data[6] = si.splev(tn, tck, der=3)

            if verbose:
                logger.log.info('No more refinements.')
                logger.log.info('\nTotal number of recursions: %s'
                                % (recursions - 1))
            return

    def get_curvature(self):
        """Curvature and radius of curvature of a parametric curve

        der1 is dx/dt and dy/dt at each point
        der2 is d2x/dt2 and d2y/dt2 at each point

        Returns:
            float: Tuple of numpy arrays carrying gradient of the curve,
                   the curvature, radiusses of curvature circles and
                   curvature circle centers for each point of the curve
        """

        coo = self.spline_data[0]
        der1 = self.spline_data[3]
        der2 = self.spline_data[4]
        der3 = self.spline_data[6]

        xd = der1[0]
        yd = der1[1]
        x2d = der2[0]
        y2d = der2[1]

        x3d = der3[0]
        y3d = der3[1]

        n = xd**2 + yd**2
        d = xd*y2d - yd*x2d

        # gradient dy/dx = dy/du / dx/du
        gradient = der1[1] / der1[0]
        gradient2 = der2[1] / der2[0]

        # radius of curvature
        R = n**(3./2.) / abs(d)
        # curvature
        C = d / n**(3./2.)

        # variation of curvature
        n2 = gradient2*n - 3*yd*y2d
        Cvar = n2 / n**(5./2.)

        # test with formula from Wang
        n_w = y2d
        d_w = 1 + yd**2.
        d2_w = y3d * d_w - 3 * yd * y2d**2
        C_w = n_w / d_w**(3./2.)
        C_var_w = d2_w / d_w**(5./2.)

        # TODO change sign for the top curve !! - C_w works

        # coordinates of curvature-circle center points
        xc = coo[0] - R * yd / np.sqrt(n)
        yc = coo[1] + R * xd / np.sqrt(n)

        self.curvature_data = [gradient, C, R, xc, yc, C_w, C_var_w]
        return self.curvature_data

    def get_leading_edge_radius(self):
        """Identify leading edge radius, i.e. smallest radius of
        parametric curve

        Returns:
            FLOAT: leading edge radius, its center and related contour
            point and id
        """

        radius = self.curvature_data[2]
        rc = np.min(radius)
        # numpy where returns a tuple
        # we take the first element, which is type array
        le_id = np.where(radius == rc)[0]
        # convert the numpy array to a list and take the first element
        le_id = le_id.tolist()[0]
        # leading edge curvature circle center
        xc = self.curvature_data[3][le_id]
        yc = self.curvature_data[4][le_id]
        xr, yr = self.spline_data[0]
        xle = xr[le_id]
        yle = yr[le_id]
        self.leading_edge_radius = rc

        return rc, xc, yc, xle, yle, le_id

    def analyze(self, tolerance, spline_points):

        # raw coordinates are stored as numpy array
        # np.array( (x, y) )

        # interpolate a spline through the raw contour points
        x, y = self.raw_coordinates
        self.spline(x, y, points=spline_points, degree=3)

        # refine the contour in order to meet the tolerance
        try:
            self.refine(tol=tolerance, verbose=False)
        except:
            pass

        # redo spline on refined contour
        # evaluate=True --> spline only evaluated at refined contour points
        coo, u, t, der1, der2, tck, der3 = self.spline_data
        x, y, = coo
        self.spline(x, y, points=spline_points, degree=3, evaluate=True)

        # get specific curve properties
        curve_data = self.get_curvature()

        return curve_data, self.spline_data


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(a, b, degree=False):
    """Returns the angle between
    vectors 'a' and 'b'
    """
    a_u = unit_vector(a)
    b_u = unit_vector(b)
    angle = np.arccos(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    if degree:
        angle *= 180.0 / np.pi
    return angle
