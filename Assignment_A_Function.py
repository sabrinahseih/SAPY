#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:10:22 2024

@author: sabrina
"""

# Import all the packages
# For analysis
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.diagnostic import lilliefors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy.spatial import distance
from scipy.stats import chi2

# For GUI
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QComboBox, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# Calculate Mahalanobis distance for each row
def calculate_mahalanobis(row, mean, inv_cov_matrix):
    # Convert the row to a 1D array
    row_1d = row.values  
    return distance.mahalanobis(row_1d, mean, inv_cov_matrix)

# Turn CTV into percentage
def turn_percentage(CTV):
    CTV= CTV*100
    return(CTV)

# Highlight the country in the tick and the bar
def highlight_country_tick(country_code, bar_below, bar_upper, ax):
    # Update x-tick labels and bar colors for the specified country
    xticks = ax.get_xticklabels()
    
    for i, tick in enumerate(xticks):
        if tick.get_text() == country_code:
            tick.set_fontweight("bold")
            tick.set_color("#b2182b")
            
            # Change the color of the selected country's bars
            bar_below[i].set_color("#4393c3")
            bar_upper[i].set_color("#d6604d")