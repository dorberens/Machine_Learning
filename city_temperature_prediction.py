import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df = df.loc[df["Year"].isin(range(1995, 2021)) & df["Month"].isin(range(1, 13)) & (df["Temp"] > 0)]
    df["DayOfYear"] = df['Date'].dt.dayofyear
    df["Year"] = df["Year"].astype(str)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_only_df = df.loc[df["Country"] == "Israel"]
    # first:
    fig1 = px.scatter(israel_only_df, x="DayOfYear", y="Temp", color="Year")
    fig1.update_layout(
        title="Relation between day of the year and a average daily temperature",
        xaxis_title="Day of year",
        yaxis_title="Temperature")
    fig1.show()
    # second:
    fig2 = px.bar(israel_only_df.groupby(["Month"], as_index=False).agg(std=("Temp", "std")), x="Month", y="std")
    fig2.update_layout(title="Standard Deviation of Avg. Temperature per month",
                       xaxis_title="Month",
                       yaxis_title="Standard deviation")
    fig2.show()

    # Question 3 - Exploring differences between countries
    fig3 = px.line(df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std")),
                   x="Month", y="mean", error_y="std", color="Country")
    fig3.update_layout(
        title="Differences of Mean an Standard Deviation of Temperature Between Countries Over the Years",
        xaxis_title="Month",
        yaxis_title="Mean")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    X = israel_only_df["DayOfYear"]
    y = israel_only_df["Temp"]
    israel_only_train_x, israel_only_train_y, israel_only_test_x, israel_only_test_y = split_train_test(X, y)
    k_degree_loss_array = np.ones((10,))
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit._fit(israel_only_train_x, israel_only_train_y)
        k_degree_loss_array[k - 1] = np.round(poly_fit._loss(israel_only_test_x, israel_only_test_y), 2)
        print(f"test error for polynomial model of degree {k}: {k_degree_loss_array[k - 1]}")

    fig4 = px.bar(x=list(range(1, 11)), y=k_degree_loss_array, text=k_degree_loss_array)
    fig4.update_layout(title="Test Error For k Degree Polynomial Fitting of Avg. Daily Temperature",
                       xaxis_title="Degree",
                       yaxis_title="Test Error")
    fig4.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_model_of_5_degree = PolynomialFitting(5)
    poly_model_of_5_degree._fit(X, y)
    model_error = np.ones((3,))
    other_countries = ["Jordan", "South Africa", "The Netherlands"]
    for i, country in enumerate(other_countries):
        subset_of_country = df[df["Country"] == country]
        model_error[i] = np.round(
            poly_model_of_5_degree._loss(subset_of_country["DayOfYear"], subset_of_country["Temp"]), 2)
    fig5 = px.bar(x=other_countries, y=model_error, color=other_countries, text=model_error)
    fig5.update_layout(title="Error of Polynomial Model per Country",
                       xaxis_title="Country",
                       yaxis_title="Error of Model",
                       legend_title="Country")
    fig5.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig5.show()
