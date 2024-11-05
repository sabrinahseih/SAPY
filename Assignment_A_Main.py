#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:58:27 2024

@author: anonomous
"""
#%%
"""
Background information:
    Country group: G20 countries
    Target year: 2050
    Median target household emissions ((tCO2e/cap)): 0.61
    Baseline or projected footprints: Target year
    Method of non-parametric correlation: Spearman rank correlation
"""

#%% Import packages and functions

from Assignment_A_Function import *

#%% 3. Import the provided data 
categories = pd.read_csv("categories.csv")
footprints = pd.read_csv("footprints.csv")
G20 = pd.read_csv("G20.csv", encoding='unicode_escape')

median_target = 0.61

#%% 4a) Filter the data for the relevant year, depending on the choices of the target year and baseline or projected footprints.
cat_year2050 = categories[categories["Year"] == 2050]
tot_year2050 = footprints[footprints["Year"] == 2050]

#%% 4b) Filter the data for the selected country group.
cat_year2050_G20 = cat_year2050[cat_year2050["Region"].isin(G20["iso3"])]
tot_year2050_G20 = tot_year2050[tot_year2050["Region"].isin(G20["iso3"])]

#%% 4c) Reshape the categories data from a long to a wide format so that the emissions of the different categories are in separate columns. This can facilitate later analysis of the data.
cat_year2050_G20.columns
cat_year2050_G20_wide = cat_year2050_G20.pivot_table(
    index = "Region",
    values = "Emissions (tCO2e cap-1)",
    columns="Category"
    )

#%% 4d) Set the region as the index in both data sets and remove the year that became redundant after the filtering.
tot_year2050_G20 = tot_year2050_G20.drop(columns="Year")
tot_year2050_G20 = tot_year2050_G20.set_index("Region")

#%% 5a) Calculate the gaps between the lifestyle carbon footprints and the emissions compatible with the 1.5°C target
# Create a new dataset that substract the lifestyle footprint with the target to see the gap
cat_year2050_G20_wide["Emission Gap"] = tot_year2050_G20["Emissions (tCO2e cap-1)"] - median_target

#%% 5b) Calculate the contribution to variance (CTV) to check how much each consumption category contributes to the variation in the gaps.

ctv_df = calculate_ctv(cat_year2050_G20_wide)

#%% 5c) Identify the category with the largest contribution to variance.

largest_ctv_index = ctv_df["CTV"].idxmax()
largest_ctv = ctv_df.loc[largest_ctv_index, "CTV"]
print("The category of the largest CTV is:", largest_ctv_index, 
      ",with the value of ", largest_ctv)

#%% 5f) Test the normality of both datasets by performing modified Kolmogorov-Smirnov / Lilliefors tests

p_value_1 = lilliefors(cat_year2050_G20_wide[largest_ctv_index],"norm")

print(f"KS Test for the category with the largest ctv, {largest_ctv_index}, against Normal Distribution:")
print(f"Statistic: {p_value_1[0]}, p-value: {p_value_1[1]}")

p_value_2 = lilliefors(cat_year2050_G20_wide["Emission Gap"],"norm")

print("KS Test for the category with the emission gap against Normal Distribution:")
print(f"Statistic: {p_value_2[0]}, p-value: {p_value_2[1]}")

#%% 5g) Test homoscedasticity by inspecting a scatter plot

plt.scatter(cat_year2050_G20_wide["Direct emissions"], cat_year2050_G20_wide["Emission Gap"])
plt.title("Direct Emissions & Emission Gaps")
plt.show()

# homoscedasticity = input("Is the scatter plot showing patterns of homoscedasticity? (Y/N)")
homoscedasticity = "Y"

if homoscedasticity == "Y":
    True
else:
    False

#%% 5h) Test the absence of bivariate outliers with the Mahalanobis distance

# Calculate Mahalanobis distance for each row
input_mahalanobis = cat_year2050_G20_wide[["Direct emissions", "Emission Gap"]]
mean = np.mean(input_mahalanobis, axis=0)
cov_matrix = np.cov(input_mahalanobis, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Apply the function to each row
input_mahalanobis["Mahalanobis_Distance"] = input_mahalanobis.apply(
    calculate_mahalanobis, axis=1, args=(mean, inv_cov_matrix)
)

# Set the threshold using Chi-squared distribution with 95% confidence
threshold = chi2.ppf(0.95, df=2)  # df=2 because we have two variables

# Identify outliers (Mahalanobis distance greater than the threshold)
input_mahalanobis["Outlier"] = input_mahalanobis["Mahalanobis_Distance"] > threshold


# View the results
"""print(input_mahalanobis[
    ["Direct emissions", "Emission Gap", "Mahalanobis_Distance", "Outlier"]])"""

outliers = input_mahalanobis[input_mahalanobis["Outlier"]]

if not outliers.empty:
    # If there are any outliers, print their index (region) and the outlier result
    for index, row in outliers.iterrows():
        print(f"Region: {index}, Outlier: {row['Outlier']}")
else:
    # If no outliers, print a message indicating that
    print("No outliers detected in any region.")

#%% 5d) Make any choices in the following subtasks through conditional statements.
#   5e) Print the conclusions from the assumption checks and the main test
# If both the largest CTV category emission and the emission gap datasets are normally distributed, 
# show homoscedasticity and have no outliers, we can choose a parametric test; 

if p_value_1[1] > 0.05 and p_value_2[1] > 0.05 and homoscedasticity & outliers.empty:
    print("The data sets are normally distributed, homoscedasticity being tested and with no outliers, thus is suitable to use a parametric test.")
else:
    print("At least one of the assumptions of normality, homoscedasticity or absence of bivariate outliers is violated, thus is suitable to use a non-parametric test.")

#%% 5i) Use the above information to choose a parametric (Pearson) or non-parametric correlation coefficient
# Calculate the correlation coefficient and its p-value, and decide whether the result is statistically significant

coefficient_correlation, pvalue = spearmanr(cat_year2050_G20_wide["Direct emissions"], 
                                            cat_year2050_G20_wide["Emission Gap"])
print(f"The coeffient correlation from the Pearson test is: {coefficient_correlation}, with a p-value of {pvalue}. The relationship between Direct Emissions and the Emission Gap seems to be linear.")
# If the coefficient correlation value is close to 1, it means it's likely to be linears

#%% 6a) Remove any categories without contributions to variance before the export.
ctv_df = ctv_df.dropna()

#%% 6b) Export the contributions to variance as percentages.

ctv_df["CTV"] = pd.to_numeric(ctv_df.CTV, errors='coerce')

print(ctv_df.dtypes)

ctv_df["CTV_%"] = ctv_df["CTV"].apply(turn_percentage)

#%% 6c) Indicate the choices made for this analysis

# Setups the template that the client requested with 3-5 rows of information 
# Followed by 1 blank rows and the dataframe
information = """\
Background information: 
    Country group: G20 countries. 
    Target year: 2050. 
    Median target household emissions ((tCO2e/cap)): 0.61
    Baseline or projected footprints: Target year
    Method of non-parametric correlation: Spearman rank correlation
{}"""
    
with open("Assignment A CTV Hsuan Hsieh.csv", "w", encoding='utf-8') as f:
    f.write(information.format("") + "\n")  # Write the information followed by a newline

# Append the DataFrame 
ctv_df.to_csv("Assignment A CTV Hsuan Hsieh.csv", mode='a', encoding='utf-8', index=False)

#%% 7a) Create a stacked bar plot that distinguishes the parts of the lifestyle carbon footprints below and above the 1.5°C target.

plot_df = cat_year2050_G20_wide[["Emission Gap"]].merge(
    tot_year2050_G20[["Emissions (tCO2e cap-1)"]], left_index=True, 
    right_index=True)
plot_df["Below target"] = median_target

fig, ax = plt.subplots()
plot_df = plot_df.sort_values(by=['Emission Gap'], ascending=True)

# Create the stacked bar chart
bar_below = ax.bar(plot_df.index.values, plot_df["Below target"], color="#92c5de")
bar_upper = ax.bar(plot_df.index.values, plot_df["Emissions (tCO2e cap-1)"], 
                   bottom=plot_df["Below target"], color="#f4a582")

# Add the horizontal line for the 1.5°C target
ax.axhline(y=0.61, color="#053061", linestyle="dashed", label="Median Emission Target: 0.61")

# Add axis titles
ax.set_xlabel("G20 Regions")
ax.set_ylabel("Emissions (tCO2e)")

# Insert project logo
import matplotlib.image as image
im = image.imread("lifestyles_logo.png")
newax = fig.add_axes([0.13, 0.77, 0.1, 0.1], zorder=11)
newax.imshow(im)
newax.set_axis_off()

# Define x-ticks and labels for countries
ax.set_xticks(range(len(plot_df.index.values)))
ax.set_xticklabels(plot_df.index.values)

# Add values
annotate_bars(ax, bar_upper, plot_df["Emissions (tCO2e cap-1)"])

#%% 7c) Highlight a country of special interest to you, e.g., by presenting it in bold.

# Highlight a specific country on x-axis and bars
highlight_country_tick("AU", bar_below, bar_upper,ax)

# Add legend 
ax.legend(frameon=False, loc="center left")

plt.show()

fig.savefig('stackbar.png', format='png', dpi=150, bbox_inches='tight')


#%% 7e) In a separate figure, create a pie chart based on the contributions to variance, also displaying the percentages of these different categories.
fig, ax = plt.subplots()

# Choose the symbology carefully and change the default colours
colors = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']
ctv_df = ctv_df.sort_values(by=['CTV'], ascending=False)
plt.pie(ctv_df["CTV_%"], startangle=90, colors=colors)

# Add a legend where relevant
handles = [f"{label}: {value:.2f}%" for label, value in zip(ctv_df.index.values, ctv_df["CTV_%"])]
plt.legend(handles, loc="center left", bbox_to_anchor=(0.95, 0.5), frameon=False)

im = image.imread("lifestyles_logo.png")
newax = plt.axes([0.13, 0.77, 0.1, 0.1], zorder=11)
newax.imshow(im)
newax.set_axis_off()

# Adjust the figure margins to reduce whitespace
plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
plt.show()

# Save the figure without large margins
fig.savefig('piechart.png', format='png', dpi=150, bbox_inches='tight')

#%% 8 Create a GUI element for user interaction
#   8a) Create a dropdown menu to choose a country that will subsequently be highlighted in the stacked bar plot.
#   8c) Add an OK button that closes the pop-up window.
#   8b) Use the country selected above as the default choice.

# G20 data removing country without data
G20_drop = G20.drop(G20[G20["country"] == "Saudi Arabia"].index)
G20_drop = G20_drop.drop(G20_drop[G20_drop["country"] == "Argentina"].index)

G20_country = G20_drop["country"].tolist()
G20_iso3 = G20_drop["iso3"].tolist()

class PlotCanvas(FigureCanvas):
    def __init__(self, country_code, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.plot(country_code)

    def plot(self, country_code):
        # Sample data for plotting
        plot_df = cat_year2050_G20_wide[["Emission Gap"]].merge(
            tot_year2050_G20[["Emissions (tCO2e cap-1)"]], left_index=True, right_index=True
        )
        plot_df["Below target"] = median_target
        plot_df = plot_df.sort_values(by=['Emission Gap'], ascending=True)

        self.ax.clear()  # Clear previous plot

        # Create stacked bar chart
        bar_below = self.ax.bar(plot_df.index.values, plot_df["Below target"], color="#92c5de")
        bar_upper = self.ax.bar(plot_df.index.values, plot_df["Emissions (tCO2e cap-1)"], bottom=plot_df["Below target"], color="#f4a582")

        # Add horizontal line for the 1.5°C target
        self.ax.axhline(y=0.61, color="#053061", linestyle="dashed")

        # Add axis titles
        self.ax.set_xlabel("G20 Regions")
        self.ax.set_ylabel("Emissions (tCO2e)")
        
        # Add values for emission gap
        for index, rect in enumerate(bar_upper):
            value = plot_df["Emissions (tCO2e cap-1)"].iloc[index]
            rounded_value = round(value, 1)
            self.ax.text(rect.get_x() + rect.get_width() / 2, 
                    rect.get_y() + rect.get_height() + 0.1,  # Positioning above the bar
                    str(rounded_value),
                    ha="center", va="bottom")

        # Highlight selected country
        xticks = self.ax.get_xticklabels()
        for i, tick in enumerate(xticks):
            if plot_df.index[i] == country_code:
                tick.set_fontweight("bold")
                tick.set_color("#b2182b")
                bar_below[i].set_color("#4393c3")
                bar_upper[i].set_color("#d6604d")

        self.draw()  # Redraw the canvas with updated plot

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Country Specific Bar Plot'
        self.left = 100
        self.top = 100
        self.width = 700
        self.height = 500
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        # Set window to stay on top and bring it to the front
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.raise_()
        self.activateWindow()

        # Create a combo box and add items
        self.combobox = QComboBox()
        self.combobox.addItems(G20_country)
        self.combobox.setFixedWidth(self.width)
        self.combobox.setCurrentText("Australia")
        self.combobox.currentIndexChanged.connect(self.update_plot)
        

        # Create an "OK" button
        ok_button = QPushButton("OK", self)
        ok_button.clicked.connect(self.close)
        ok_button.setFixedWidth(self.width)

        # Initialize plot with default country (AU)
        self.canvas = PlotCanvas(country_code="AU", parent=self)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.combobox)
        layout.addWidget(self.canvas)
        layout.addWidget(ok_button)
        layout.setAlignment(self.combobox, QtCore.Qt.AlignCenter)
        layout.setAlignment(ok_button, QtCore.Qt.AlignCenter)

        # Create a container widget and set the layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.show()

    def update_plot(self):
        # Get the selected country's ISO code
        selected_country = self.combobox.currentText()
        country_code = G20_iso3[G20_country.index(selected_country)]

        # Update the plot in the canvas with the new country code
        self.canvas.plot(country_code)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    sys.exit(app.exec_())
    
#%% Optimization
import cProfile
import pstats

# Define the slow function
def calculate_ctv(cat_year2050_G20_wide):
    # Extract the "Emission Gap" column once to avoid repeated access
    emission_gap = cat_year2050_G20_wide["Emission Gap"]
    
    # Pre-compute the columns to iterate over, excluding "Emission Gap"
    columns_to_iterate = [col for col in cat_year2050_G20_wide.columns if col != "Emission Gap"]
    
    ctv_list = []
    
    # Calculate the correlation squared values for each category
    for category in columns_to_iterate:
        # Calculate the Spearman correlation with the emission gap
        correlation, _ = spearmanr(emission_gap, cat_year2050_G20_wide[category])
        
        # Only append if the correlation is not NaN or undefined
        if not pd.isna(correlation):
            ctv_list.append({"Category": category, "Correlation": correlation})
    
    # Convert the list to a DataFrame
    ctv_df = pd.DataFrame(ctv_list)
    if not ctv_df.empty:
        # Calculate Correlation Squared and CTV
        ctv_df["Correlation Squared"] = ctv_df["Correlation"] ** 2
        total_correlation_squared = ctv_df["Correlation Squared"].sum()
        ctv_df["CTV"] = ctv_df["Correlation Squared"] / total_correlation_squared
        return ctv_df.set_index("Category")
    else:
        return pd.DataFrame()
    
# Create a profiler
pr = cProfile.Profile()
pr.enable()  # Start profiling

# Call the function you want to profile
cat_year2050_G20_wide["Emission Gap"] = tot_year2050_G20["Emissions (tCO2e cap-1)"] - median_target
calculate_ctv(cat_year2050_G20_wide)  # Ensure this function is defined

pr.disable()  # Stop profiling

# Create stats object to sort and print results
ps = pstats.Stats(pr).strip_dirs().sort_stats('cumulative')
ps.print_stats(10)  # Print the top 10 results



