import numpy as np
import bezier
import splipy as sp
from shapely.geometry import Polygon
from lib.contour_analysis import ContourAnalysis
from scipy.optimize import minimize
import os
from csaps import csaps
from csaps import CubicSmoothingSpline
import matplotlib.pyplot as plt
import time
import matplotlib.style as style
import copy

from lib.airfoil_parametrisation import AirfoilGeometry
np.seterr(divide='ignore', invalid='ignore')

airfoil_filename = '../airfoils_for_testing/LA5055_fit.dat'
# airfoil_filename = '../airfoils/Skawinski_0.txt'

spline_order = 3

test = AirfoilGeometry()
test.read_coordinates(airfoil_filename)

colors = [[0, 0, 0], '#0F95D7', [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255], [0, 158 / 255, 115 / 255],
          [230 / 255, 159 / 255, 0]]
style.use('fivethirtyeight')


# calculate the angles
test.calc_centroid()


deltax = 0.5 - test.x
deltaz = 0.0 - test.z
angles = np.arctan(deltaz/deltax)
angles *= 180/np.pi

# use arclength instead
ds = np.zeros(len(test.x))
for cntr in range(len(test.x)-1):
    dx = test.x[cntr+1] - test.x[cntr]
    dz = test.z[cntr+1] - test.z[cntr]
    ds[cntr+1] = np.sqrt(dx**2 + dz**2)

arclength = np.cumsum(ds)


nr_pts = 251


original = copy.deepcopy(test)

test.refine_coordinates_no_scaling(nr_pts=nr_pts, spline_order=spline_order)

# test.interpolate_coordinates(nr_pts=nr_pts, order=spline_order)

fig, ax = plt.subplots(figsize=(14, 5))
plt.plot(original.x, original.z, fillstyle='none', marker='o', markerfacecolor='none',
         markeredgecolor=colors[0], linestyle='none', markersize='10', markeredgewidth='2')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax1 = plt.plot(test.x, test.z, '#0F95D7')
plt.plot(test.x, test.z, fillstyle='none', marker='o', markerfacecolor='none',
         markeredgecolor='#0F95D7', linestyle='none', markersize='10', markeredgewidth='2')
plt.show()


test.calculate_curvature()
test.calc_trailing_edge_angle()
test.calc_area()

theta = np.linspace(0.0, 2.0 * np.pi, nr_pts)
x = 0.5 * (np.cos(theta) + 1.0)
x_vec = x[0:int(np.floor(nr_pts / 2)) + 1]


curv_bot,curv_top = test.local_curvature(x_over_c=x_vec)
curv_var_bot,curv_var_top = test.local_curvature_variation(x_over_c=x_vec)

test.calc_number_of_reversals()


fig,ax = plt.subplots(figsize=(14, 5))
plt.plot(x_vec, np.sign(curv_top)*np.float_power(np.abs(curv_top),0.3), '#0F95D7')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

fig,ax = plt.subplots(figsize=(14, 5))
plt.plot(x_vec, np.sign(curv_bot)*np.float_power(np.abs(curv_bot),0.3),'#0F95D7')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

fig,ax = plt.subplots(figsize=(14, 5))
plt.plot(x_vec, curv_var_top, '#0F95D7')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

fig,ax = plt.subplots(figsize=(14, 5))
plt.plot(x_vec, curv_var_bot, '#0F95D7')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

fig,ax = plt.subplots(figsize=(14, 5))
plt.plot(test.x_curvature, test.gradient, '#0F95D7')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()


test.calc_thickness_and_camber()

print("max thickness = ", test.max_thickness)
print("max camber = ", test.max_camber)
print("x max thickness = ", test.x_max_thickness)
print("x max camber = ", test.x_max_camber)
print("leading edge radius = ", test.leading_edge_radius)
print("trailing edge angle = ", test.trailing_edge_angle)
print('airfoil area', test.area)
print('nr bottom reversals', test.nr_bottom_reversals)
print('nr top reversals', test.nr_top_reversals)

# now work out the derivatives, curvatures, ...
fig, ax = plt.subplots(figsize=(14, 5))
plt.plot(original.x, original.z, fillstyle='none', marker='o', markerfacecolor='none',
         markeredgecolor=colors[0], linestyle='none', markersize='10', markeredgewidth='2')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax1 = plt.plot(test.x,test.z, '#0F95D7')
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
plt.plot(test.x_thickness, test.thickness, '#0F95D7')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
plt.plot(test.x_thickness, test.camber, '#0F95D7')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()

test.write_coordinates_to_file('../airfoils_for_testing/test.dat')


