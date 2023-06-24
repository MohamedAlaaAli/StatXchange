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

NUM_ROWS = 4
NUM_COLS = 4

def load_data(path):
    """
    Load a CSV file into a DataFrame.

    Parameters:
        path (str): The file path of the CSV file to load.

    Returns:
        pandas.DataFrame: The DataFrame containing the loaded data.

    Prints:
        - The first 10 rows of the loaded DataFrame.
        - Summary information about the loaded DataFrame using the `info()` method.

    Example:
        df = load_data('data.csv')
    """
    df = pd.read_csv(path)
    print(df.head(10))
    print(df.info())
    return df


def display_nulls(df):
    """
    Display the missing values in a DataFrame using a matrix plot.

    Parameters:
        df (pandas.DataFrame): The DataFrame to visualize.

    Displays:
        The matrix plot showing the missing values in the DataFrame.

    Example:
        display_nulls(df)
    """
    msn.matrix(df)
    plt.show()


def big_pic(df, quantitative_features):
    """
    Generate box plots for the specified quantitative features in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        quantitative_features (list): A list of column names representing the quantitative features.

    Displays:
        Box plots for the specified quantitative features.

    Example:
        quantitative_features = ['Feature1', 'Feature2', 'Feature3']
        big_pic(df, quantitative_features)
    """
    # Plotting box plots for all features
    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(df[quantitative_features])
    ax.set_xticklabels(quantitative_features, rotation='vertical', fontsize="20")
    ax.set_xlabel("Features", fontsize="20")
    ax.set_ylabel("Values", fontsize="20")
    ax.set_title("Box Plots for Quantitative Features", fontsize="25")
    plt.show()


def box_plots(df, quantitative_features):
    """
    Generate individual box plots for each quantitative feature in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        quantitative_features (list): A list of column names representing the quantitative features.

    Displays:
        Individual box plots for each quantitative feature.

    Example:
        quantitative_features = ['Feature1', 'Feature2', 'Feature3']
        box_plots(df, quantitative_features)
    """
    # Create the subplots
    fig, axes = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(20, 30))

    # Iterate over the features and create a subplot for each feature
    for i, feature in enumerate(quantitative_features):
        # Calculate the subplot position
        row = i // NUM_ROWS
        col = i % NUM_COLS
        axes[row, col].boxplot(df[feature])
        axes[row, col].set_title(feature, fontsize="20")

    # Adjust the spacing between subplots
    fig.tight_layout()
    plt.show()


def plot_dist(row, col, df, Transform=[]):
    """
    Generate distribution plots for the columns of a DataFrame.

    Parameters:
        row (int): Number of rows in the subplot grid.
        col (int): Number of columns in the subplot grid.
        df (pandas.DataFrame): The DataFrame containing the data.
        Transform (list, optional): List of transformation functions to apply to the columns. Default is an empty list.

    Displays:
        Distribution plots for the columns of the DataFrame.

    Example:
        row = 2
        col = 3
        Transform = [np.log1p, np.sqrt]
        plot_dist(row, col, df, Transform)
    """
    # create a figure and axis
    fig, ax = plt.subplots(row, col, figsize=(40, 30))
    for i, feature in enumerate(df.columns):
        # determine the current ax position
        row = i // NUM_ROWS
        col = i % NUM_COLS
        if len(Transform) == 0:
            sns.histplot(df[feature], kde=True, ax=ax[row, col])
        else:
            sns.histplot(df[feature].agg(Transform),
                         kde=True, ax=ax[row, col])

        ax[row, col].set_title(f"Distribution of {feature}", fontsize="25")

    fig.tight_layout()
    plt.show()


def boxcox_transform(data):
    """
    Apply the Box-Cox transformation to a given dataset.

    Parameters:
        data (array-like): The input data to be transformed.

    Returns:
        array-like: The transformed data.

    Example:
        transformed_data = boxcox_transform(data)
    """
    trans, _ = boxcox(data)
    return trans


def categorical_featuers(df, categorical_features):
    """
    Generate count plots for categorical features in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        categorical_features (list): A list of column names representing the categorical features.

    Displays:
        Count plots for the categorical features of the DataFrame.

    Example:
        categorical_features = ['Feature1', 'Feature2', 'Feature3']
        categorical_features(df, categorical_features)
    """
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(
        categorical_features), figsize=(15, 5))

    # Iterate over each categorical feature and plot the countplot
    for i, feature in enumerate(categorical_features):
        sns.countplot(x=feature, data=df, ax=axes[i])
        axes[i].set_title(feature)

    # Adjust the layout and spacing between subplots and display the plot
    plt.tight_layout()
    plt.show()


# Helper method to calculate the mode of a certain feature
def calculate_mode(x):
    return x.mode().iat[0]

# Helper method to calculate the range of a certain feature


def calculate_range(y):
    return y.max() - y.min()


# Helper method to calculate the IQR of a certain feature
def calculate_IQR(y):
    return y.quantile(0.75) - y.quantile(0.25)


def measures(df, quantitative_features):
    """
    Calculate various statistical measures for quantitative features in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        quantitative_features (list): A list of column names representing the quantitative features.

    Returns:
        dict: A dictionary containing the calculated measures for each feature.

    Example:
        quantitative_features = ['Feature1', 'Feature2', 'Feature3']
        measures_dict = measures(df, quantitative_features)
    """
    msr = []
    for feature in quantitative_features:
        msr.append(df[feature].agg(
            ["mean", "median", calculate_mode, "var", "std", calculate_range, calculate_IQR]))
    for item in msr:
        item.rename({"calculate_mode": "mode", "var": "variance", "std": "standard deviation",
                     "calculate_range": "range", "calculate_IQR": "IQR"}, inplace=True)
        print(item)
        print("--------------------------------")
    return dict(zip(["mean", "median", "mode", "var", "std", "range", "iqr"],msr))


def stndrd(df, quantitative_features, disc):
    """
    Perform standardization on quantitative features in a DataFrame using a reference distribution.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to be standardized.
        quantitative_features (list): A list of column names representing the quantitative features to be standardized.
        disc (pandas.DataFrame): The reference distribution DataFrame used for standardization.

    Returns:
        pandas.DataFrame: The standardized DataFrame.

    Example:
        quantitative_features = ['Feature1', 'Feature2', 'Feature3']
        standardized_df = stndrd(df, quantitative_features, disc)
    """
    standard_df = (df[quantitative_features] - disc[quantitative_features].iloc[1, :]
                   ) / df[quantitative_features].iloc[2, :]
    return standard_df

def split(features, targets):
    """
    Splits the input data into training and testing sets using scikit-learn's `train_test_split` function.

    Parameters:
    -----------
    features : array-like, shape (n_samples, n_features)
        The input features to be split into training and testing sets.

    targets : array-like, shape (n_samples,)
        The target values to be split into training and testing sets.

    Returns:
    --------
    X_train : array-like, shape (n_train_samples, n_features)
        The training set of input features.

    X_test : array-like, shape (n_test_samples, n_features)
        The testing set of input features.

    y_train : array-like, shape (n_train_samples,)
        The training set of target values.

    y_test : array-like, shape (n_test_samples,)
        The testing set of target values.

    Raises:
    -------
    AssertionError:
        If the number of samples in `X_train` and `y_train` is not equal.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, random_state=0, test_size=0.2)
    assert X_train.shape[0] == y_train.shape[0]

    return X_train, X_test, y_train, y_test

def plot_cond(df):
    """
    Plot conditional distributions of each feature in a DataFrame based on the "death" column.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None

    Example:
        plot_cond(df)
    """
    for col in df.drop("death", axis=1).columns:
        sns.displot(data=df, x=col, col="death", kde=True)
        plt.show()


def violin(features, df):
    """
       Create violin plots for specified features in a DataFrame, with additional grouping and splitting based on the "Race" and "Sex" columns.

       Parameters:
           features (list): A list of column names representing the features to create violin plots for.
           df (pandas.DataFrame): The DataFrame containing the data.

       Returns:
           None

       Example:
           features = ['Feature1', 'Feature2', 'Feature3']
           violin(features, df)
       """
    for feature in features:
        if feature == "Race" or feature == "Sex":
            continue
        g = sns.catplot(data=df, x="death", y=feature, kind="violin",
                        hue="Sex", palette="pastel", split=True, col="Race")
        plt.show()

def corr(transformed_data):
    """
    Create a correlation heatmap for a given DataFrame of transformed data.

    Parameters:
        transformed_data (pandas.DataFrame): The DataFrame containing the transformed data.

    Returns:
        None

    Example:
        corr(transformed_data)
    """
    cmap = sns.diverging_palette(300, 150, s=40, l=65, n=10)
    corrmat = transformed_data.corr()
    plt.subplots(figsize=(18, 18))
    sns.heatmap(corrmat, cmap=cmap, annot=True, square=True)
    plt.show()