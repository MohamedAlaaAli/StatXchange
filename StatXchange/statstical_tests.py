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

# Let's create our shapiro-wilk test function
def shapiro_test(data):
    """
    Perform the Shapiro-Wilk test for normality on each column of a given DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        p_vals (list): A list of p-values obtained from the Shapiro-Wilk test for each column.

    Example:
        p_values = shapiro_test(data)
    """
    lst_stats, p_vals = [], []
    for col in data.columns:
        statistic, p_val = shapiro(data)
        lst_stats.append(statistic)
        p_vals.append(p_val)
    return p_vals

def QQ(data):
    """
    Create Q-Q plots for each column in a given DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None

    Example:
        QQ(data)
    """
    columns = data.columns
    num_columns = len(columns)
    num_rows = int(np.ceil(num_columns / 4))

    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 3 * num_rows))
    axes = axes.flatten()
    for i, feature in enumerate(columns):
        ax = axes[i]
        sm.qqplot(data[feature], line='s', ax=ax)
        ax.set_title(f"Q-Q Plot for {feature}")

    # Remove any remaining empty subplots
    for j in range(num_columns, num_rows * 4):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# First, let's write the Anderson-Starling test function
def anderson_test(data):
    """
    Perform the Anderson-Darling test for normality on each column of a given DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        alpha_list (list): A list of significance levels obtained from the Anderson-Darling test for each column.
        stats_list (list): A list of test statistics obtained from the Anderson-Darling test for each column.
        critical_vals_list (list): A list of critical values obtained from the Anderson-Darling test for each column.

    Example:
        alpha_list, stats_list, critical_vals_list = anderson_test(data)
    """
    alpha_list, stats_list, critical_vals_list = [], [], []
    for col in data.columns:
        result = anderson(data[col])
        stats_list.append(result.statistic)
        critical_vals_list.append(result.critical_values)
        alpha_list.append(result.significance_level)
    return alpha_list, stats_list, critical_vals_list


