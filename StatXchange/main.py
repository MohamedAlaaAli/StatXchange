import functions
import handel_outliers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.model_selection import train_test_split
import missingno as msn
import math
from scipy.stats import boxcox
from scipy.stats import tstd
from scipy.stats import shapiro
import statsmodels.api as sm
from scipy.stats import anderson
df = functions.load_data("Data.csv")

# Our categorical features
categorical_features = ["Race", "Sex", "death"]
print(f"Categorical features are: {[c for c in categorical_features]}")

# Our quantitative features
quantitative_features = [
    q for q in df.columns if q not in categorical_features]
print(f"Quantitative features are: {quantitative_features}")

# Checking whether we have any quantitative zero values or not
for feature in df.columns:
    print(f"{feature} : {df[feature].eq(0).any()}")

# search for nulls
functions.display_nulls(df)

# Checking whether we have any duplicate rows or not
df.duplicated().any()

# Plotting box plots for all features
functions.big_pic(df, quantitative_features)

# plot each feature as a box plot
functions.box_plots(df, quantitative_features)

# remove outliers with IQR method
filtered_data = handel_outliers.remove_outliers_iqr(df, quantitative_features)
# visualizing after outliers removal
functions.box_plots(filtered_data, quantitative_features)
# First, we will visualize the feature distribution without any transformations
functions.plot_dist(5, 4, df)
# Let's visualize the square root transformation vs logarithmic transformation
functions.plot_dist(4, 4, df[quantitative_features], Transform=["sqrt", "log"])

# Let's copy our original DF, and apply Box-Cox transformation to the new DF
boxcox_df = df.copy(deep=True)
for feature in quantitative_features:
    boxcox_feature = functions.boxcox_transform(df[feature])
    boxcox_df[feature] = boxcox_feature
# Let's have summary statistic about our Box-Cox transformed data
disc = boxcox_df.describe()
# Take a look at the quantitative features distributions of the Box-Cox transformed data
functions.plot_dist(4, 4, boxcox_df[quantitative_features])

# Now, let's visualize the distributions of our categorical features
functions.categorical_featuers(df, categorical_features)

# Let's calculate our measures of central tendency and dispersion
stats = functions.measures(boxcox_df, quantitative_features)

# Let's apply the standardization method to our dataframe
standard_df = functions.stndrd(boxcox_df, categorical_features, disc)
# Take a look at the data after standardization
standard_df.head()
# copying box_cox data splitting the data to features and target series
transformed_data = boxcox_df.copy(deep=True)
features = transformed_data.drop("death", axis=1)
targets = transformed_data["death"]
print(f"Target features shape: {targets.shape}")
print(f"Features shape: {features.shape}")

# visualizing the conditional distributions
functions.plot_cond(transformed_data)

# splitting the data into training and testing data
X_train, X_test, y_train, y_test = functions.split(features, targets)

