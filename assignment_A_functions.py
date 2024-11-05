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
    """
    Calculate Correlation to Variance (CTV) for emissions categories.
    
    This function computes the Spearman correlation between each emissions category
    and the emission gap, then calculates the CTV for each category as a percentage 
    of the total correlation squared.

    Args:
        cat_year2050_G20_wide (pd.DataFrame): A DataFrame with emissions categories 
            as columns, including "Emission Gap".

    Returns:
        pd.DataFrame: A DataFrame with CTV values indexed by category.

    Examples:
        >>> calculate_ctv(cat_year2050_G20_wide)
        Category  Correlation  Correlation Squared  CTV
        Cat1      0.45        0.2025               0.345
        Cat2      0.78        0.6084               0.655
    """
    
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
    """
    Calculate Mahalanobis distance for a given row.
    
    This function computes the Mahalanobis distance between a given row and the
    specified mean using the provided inverse covariance matrix.

    Args:
        row (pd.Series): A row of data for which to calculate the distance.
        mean (np.array): The mean vector of the distribution.
        inv_cov_matrix (np.array): The inverse covariance matrix of the distribution.

    Returns:
        float: The Mahalanobis distance of the row from the mean.

    Examples:
        >>> calculate_mahalanobis(row, mean, inv_cov_matrix)
        2.45
    """
    # Convert the row to a 1D array
    row_1d = row.values  
    return distance.mahalanobis(row_1d, mean, inv_cov_matrix)

# Turn CTV into percentage
def turn_percentage(CTV):
    """
    Convert CTV values to percentages.
    
    This function multiplies each CTV value by 100 to convert it from a proportion 
    to a percentage.

    Args:
        CTV (float): The Correlation to Variance (CTV) value as a proportion.

    Returns:
        float: The CTV value as a percentage.

    Examples:
        >>> turn_percentage(0.345)
        34.5
    """
    CTV= CTV*100
    return(CTV)

def create_emission_gap_plot(ax, cat_year2050_G20_wide, tot_year2050_G20, median_target, country_code):
    """
    Creates a stacked bar chart of the emission gap for G20 regions.
    
    This function visualize the emission gap for G20 regions in a stacked bar with an indication of emission target.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        cat_year2050_G20_wide (DataFrame): DataFrame containing emission gap data.
        tot_year2050_G20 (DataFrame): DataFrame containing total emissions data.
        median_target (float): The median target value for emissions.
        country_code (str): The ISO3 code of the country to highlight.

    Returns:
        None
    """
    # Prepare data for plotting
    plot_df = cat_year2050_G20_wide[["Emission Gap"]].merge(
        tot_year2050_G20[["Emissions (tCO2e cap-1)"]], left_index=True, right_index=True)
    plot_df["Below target"] = 0.61
    plot_df = plot_df.sort_values(by=["Emission Gap"], ascending=True)

    # Clear the axes before plotting
    ax.clear()

    # Create the stacked bar chart
    bar_below = ax.bar(plot_df.index.values, plot_df["Below target"], color="#92c5de", label="Below Target")
    bar_upper = ax.bar(
        plot_df.index.values,
        plot_df["Emissions (tCO2e cap-1)"],
        bottom=plot_df["Below target"],
        color="#f4a582",
        label="Above Target",
    )

    # Add the horizontal line for the 1.5°C target
    ax.axhline(y=0.61, color="#053061", linestyle="dashed", label="Median Emission Target: 0.61")

    # Add axis titles
    ax.set_xlabel("G20 Regions")
    ax.set_ylabel("Emissions (tCO2e)")

    # Define x-ticks and labels for countries
    ax.set_xticks(range(len(plot_df.index.values)))
    ax.set_xticklabels(plot_df.index.values)

    # Annotate values on the bars
    annotate_bars(ax, bar_upper, plot_df["Emissions (tCO2e cap-1)"])

    # Highlight the specified country
    highlight_country_tick(country_code, bar_below, bar_upper, ax)

    # Add legend
    ax.legend(frameon=False, bbox_to_anchor=(0.7, 1))

    # Add a project logo if needed
    im = image.imread("lifestyles_logo.png")
    newax = ax.figure.add_axes([0.13, 0.77, 0.1, 0.1], zorder=11)
    newax.imshow(im)
    newax.set_axis_off()

# Annotate bar values
def annotate_bars(ax, bar_container, values, y_offset=0.1, round_digits=1):
    """
    Annotate bar chart with values above each bar.
    
    This function adds numerical annotations to a bar chart for clarity.

    Args:
        ax (matplotlib.axes.Axes): The axes object containing the bar chart.
        bar_container (list): A list of matplotlib bar objects (e.g., from ax.bar()).
        values (pd.Series): The values to annotate on each bar.
        y_offset (float, optional): The vertical offset for the text annotations. 
            Default is 0.1.
        round_digits (int, optional): The number of decimal places to round the values. 
            Default is 1.

    Returns:
        None

    Examples:
        >>> annotate_bars(ax, bar_container, values)
        >>> # Custom offset and rounding
        >>> annotate_bars(ax, bar_container, values, y_offset=0.2, round_digits=2)
    """
    for index, rect in enumerate(bar_container):
        value = round(values.iloc[index], round_digits)
        ax.text(rect.get_x() + rect.get_width() / 2, 
                rect.get_y() + rect.get_height() + y_offset,  # Positioning above the bar
                str(value),
                ha="center", va="bottom")

# Highlight the country in the tick and the bar
def highlight_country_tick(country_code, bar_below, bar_upper, ax):
    """
    Highlight the specified country in bar chart ticks and bars.
    
    This function makes the specified country's tick label bold and changes the 
    color of the corresponding bars for emphasis.

    Args:
        country_code (str): The ISO3 code of the country to highlight.
        bar_below (list): List of bar objects for the lower bars (e.g., from ax.bar()).
        bar_upper (list): List of bar objects for the upper bars (e.g., from ax.bar()).
        ax (matplotlib.axes.Axes): The axes object containing the bar chart.

    Returns:
        None

    Examples:
        >>> highlight_country_tick("USA", bar_below, bar_upper, ax)
    """
    # Update x-tick labels and bar colors for the specified country
    xticks = ax.get_xticklabels()
    
    for i, tick in enumerate(xticks):
        if tick.get_text() == country_code:
            tick.set_fontweight("bold")
            tick.set_color("#b2182b")
            
            # Change the color of the selected country's bars
            bar_below[i].set_color("#4393c3")
            bar_upper[i].set_color("#d6604d")