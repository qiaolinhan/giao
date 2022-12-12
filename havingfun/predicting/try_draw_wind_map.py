#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: try01.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-12-10
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:try to plot a wind map
    # # For the vector arrow pointer:
    # quiver = axe.quiver(x, y, u, v, # Longitude, latitude, v_wind slides logitude, w_wind slides latitude 
    #                     pivot = 'tip', # The position of the arrow, 'tail', 'mid', 'tip'  
    #                     width = 0.01,  # The width of the arrow
    #                     scale = 10,  # The scale of the arrow, decrease the number to enlarge the scale
    #                     color = 'red', # The color of the arrow
    #                     headwidth = 4, # The headwidth of the arrow
    #                     alpha = 1, # The transparent rate of the arrow
    #                     transform = ccrs.PlateCarree(), # Could be same as the type of figure
    # )

# To note the arrows
    # axe.quiverkey(quiver, # To specify the clase of quiver
    #               x, y, # Related to the coordinates, the coorditates could be set as {'axes', 'figure', 'data', 'inches'} 
    #               coordinates = 'axes', 
    #               U, "U text", # The number of wind speed, type: int, float
    #               labelpos = 'E', # labelpos, {'N', 'S', 'E', 'W'}
    #               fontproperties = {'size': 10, 'family': 'Times New Roman'}
    # )
# ------------------------------------------------------------------------------

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from matplotlib.font_manager import FontProperties
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter 
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER 
import matplotlib.ticker as mticker
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from wrf import getvar, to_np
import cmaps

# To show 'Chinese correctly'
Simsun = FontProperties(fname = "./font/Simsun.ttf")
Times = FontProperties(fname = "./font/times.ttf")
config = {
        "font.family": 'serif',
        "mathtext.fontset": 'stix',
        "font.serif": [],
        }
mpl.rcParams.update(config)

# To show '-' correctly
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize = (5,5), dpi = 150)
axe = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
axe.set_title('$\mathrm{Wind\ Map}$', fontsize = 12, y = 1.05)

axe.add_feature(cfeat.COASTLINE.with_scale('10m'), linewidth = 0.5, color = 'k')
LAKES_border = cfeat.NaturalEarthFeature('pysical', 'lakes', '10m', edgecolor = 'k', facecolor = 'never')
axe.add_feature(LAKES_border, linewidth=0.8)
# to add more features
axe.add_feature(cfeat.LAND.with_scale('10m'))
axe.add_feature(cfeat.OCEAN.with_scale('10m'))
axe.add_feature(cfeat.RIVERS.with_scale('10m'))

# The plot area of the map
axe.set_extent([-75, -68, 44, 47], crs = ccrs.PlateCarree())
gl = axe.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.1, color = 'gray', linestyle = ':')
# gl.top_labels, gl.bottom_labels, gl.right_labels, gl.left_labels = False, False, False, False
gl.xlocator = mticker.FixedLocator(np.arange(-74.7, -68.1, 0.1))
gl.ylocator = mticker.FixedLocator(np.arange(44.9, 46.4, 0.1))
axe.set_xticks(np.arange(-75, -68, 0.1), crs = ccrs.PlateCarree())
axe.set_yticks(np.arange(44, 47, 0.1), crs = ccrs.PlateCarree())
axe.xaxis.set_major_formatter(LongitudeFormatter())
axe.yaxis.set_major_formatter(LatitudeFormatter())
axe.tick_params(labelcolor = 'k', length = 2)
labels = axe.get_xticklabels() + axe.get_yticklabels()
[label.set_fontproperties(FontProperties(fname = "./font/Times.ttf", size = 8)) for label in labels]

ncfile = nc.Dataset('~/dev/giao/data/prediction_use/POWER_Point_Daily_20210101_20210301_045d1537N_073d7012W_LST.nc')
ua = getvar(ncfile, 'ua', timeidx = 126)[0, :, :]
va = getvar(ncfile, 'va', timeidx = 126)[0, :, :]
lat = getvar(ncfile, 'lat')
lon = getvar(ncfile, 'lon')
quiver = axe.quiverkey(quiver, 0.91, 1.03, 3, "3m/s", 
                       labelpos = "E", coordinates = "axes", 
                       fontproperties = {"size": 10, "family":"Times New Roman"})
plt.show()
