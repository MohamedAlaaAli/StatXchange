# Importing packages we will be using
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

def remove_outliers_iqr(df, quantitative_features):
    """
    Remove outliers from the quantitative features of a DataFrame using the interquartile range (IQR) method.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        quantitative_features (list): A list of column names representing the quantitative features.

    Returns:
        pandas.DataFrame: The DataFrame with outliers removed.

    Example:
        quantitative_features = ['Feature1', 'Feature2', 'Feature3']
        df = remove_outliers_iqr(df, quantitative_features)
    """
    def remove_outliers_iqr(data, column):
        # Calculate the first quartile and the third quartile
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        # Calculate the IQR
        iqr = q3 - q1
        # Calculating the thrshold
        threshold = 1.5 * iqr
        # Defining the lower and upper bounds
        lower_bound = q1 - threshold
        upper_bound = q3 + threshold
        # Filtering and returning the filtered data
        filtered_data = data[(data[column] >= lower_bound) &
                             (data[column] <= upper_bound)]
        return filtered_data
    for q in df[quantitative_features]:
        df = remove_outliers_iqr(df, q)

    return df

