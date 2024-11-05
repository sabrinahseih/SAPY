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

# Calculate CTV
def calculate_ctv(cat_year2050_G20_wide):
    ctv_list = []
    
    # Calculate the correlation squared values for each category
    for category in cat_year2050_G20_wide.columns:
        if category != "Emission Gap":  # Skip the emission gap itself
            # Calculate the Spearman correlation with the emission gap
            correlation, _ = spearmanr(cat_year2050_G20_wide["Emission Gap"], 
                                       cat_year2050_G20_wide[category])
            # Append the correlation squared to the list
            ctv_list.append({"Category": category, "Correlation": correlation})
    
    # Create DataFrame and calculate CTV
    ctv_df = pd.DataFrame(ctv_list)
    ctv_df["Correlation Squared"] = ctv_df["Correlation"] ** 2
    total_correlation_squared = ctv_df["Correlation Squared"].sum()
    ctv_df["CTV"] = ctv_df["Correlation Squared"] / total_correlation_squared
    return ctv_df.set_index("Category")

# Calculate Mahalanobis distance for each row
def calculate_mahalanobis(row, mean, inv_cov_matrix):
    # Convert the row to a 1D array
    row_1d = row.values  
    return distance.mahalanobis(row_1d, mean, inv_cov_matrix)

# Turn CTV into percentage
def turn_percentage(CTV):
    CTV= CTV*100
    return(CTV)

# Annotate bar values
def annotate_bars(ax, bar_container, values, y_offset=0.1, round_digits=1):
    """
    Annotates bar chart with values above each bar.
    
    Parameters:
    - ax: matplotlib Axes object where bars are drawn.
    - bar_container: List of matplotlib bar objects (e.g., the result of ax.bar()).
    - values: List or pandas Series of values to display on the bars.
    - y_offset: Float, vertical offset above each bar for text placement.
    - round_digits: Integer, number of decimal places to round the values.
    """
    for index, rect in enumerate(bar_container):
        value = round(values.iloc[index], round_digits)
        ax.text(rect.get_x() + rect.get_width() / 2, 
                rect.get_y() + rect.get_height() + y_offset,  # Positioning above the bar
                str(value),
                ha="center", va="bottom")

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