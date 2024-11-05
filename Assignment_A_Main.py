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

from assignment_A_functions import *

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
plt.xlabel("Direct Emissions (tCO2e cap-1)")
plt.ylabel("Emission Gap (tCO2e cap-1)")
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

# Background information setup
information = """\
Background information: 
    Country group: G20 countries. 
    Target year: 2050. 
    Median target household emissions ((tCO2e/cap)): 0.61
    Baseline or projected footprints: Target year
    Method of non-parametric correlation: Spearman rank correlation
{}"""
    
with open("contribution.csv", "w", encoding='utf-8') as f:
    f.write(information.format("") + "\n") 

# Append the DataFrame 
ctv_df.to_csv("contribution.csv", mode='a', encoding='utf-8', index=False)

#%% 7a) Create a stacked bar plot that distinguishes the parts of the lifestyle carbon footprints below and above the 1.5°C target.

fig, ax = plt.subplots()
create_emission_gap_plot(ax, cat_year2050_G20_wide, tot_year2050_G20, median_target, "AU")
plt.show()

fig.savefig('stacked bar.png', format='png', dpi=150, bbox_inches='tight')


#%% 7e) Create a pie chart based on the contributions to variance, also displaying the percentages of these different categories.
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
fig.savefig('pie chart.png', format='png', dpi=150, bbox_inches='tight')

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
        self.ax.clear()  # Clear previous plot
        create_emission_gap_plot(self.ax, cat_year2050_G20_wide, tot_year2050_G20, median_target, country_code)
        self.draw() 

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
