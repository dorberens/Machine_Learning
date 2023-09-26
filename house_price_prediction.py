from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

invalid_rows_of_test_price = []
irrelevant_features = ["id", "date", "long", "lat", "sqft_lot", "sqft_lot15"]
mean_feature_dict = {}


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if y is None:
        return preprocess_test_data(X)
    else:
        return preprocess_train_data(X, y)


def preprocess_test_data(X):
    X = X.drop(invalid_rows_of_test_price, axis=0)
    X = X.drop(irrelevant_features, axis=1)
    X["renovated_recently"] = np.where(X["yr_renovated"] >= 2000, 1, 0)
    X = X.drop(["yr_renovated"], axis=1)
    for feature, mean in mean_feature_dict.items():
            X[feature].fillna(mean, inplace=True)
    X = fix_invalid_values(X)
    X["zipcode"] = X["zipcode"].astype(int)
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

    return X


def preprocess_train_data(X, y):
    df = X.assign(price=y)
    df = df.dropna().drop_duplicates()
    df.drop(irrelevant_features, axis=1, inplace=True)
    df = check_values_in_range(df)
    df["renovated_recently"] = np.where(df["yr_renovated"] > 2000, 1, 0)
    df.drop(["yr_renovated"], axis=1, inplace=True)
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode', columns=['zipcode'])

    return df.drop(["price"], axis=1), df["price"]


def check_values_in_range(df):
    df = df[(df["bedrooms"].isin(range(12))) & (df["bathrooms"] >= 0) & (df["bathrooms"] <= 8) &
            (df["sqft_living"] >= 300) & (df["sqft_living"] <= 120000) &
            (df["floors"] >= 1) & (df["floors"] <= 3.5) & (df["waterfront"].isin([0, 1])) &
            (df["view"].isin([0, 1, 2, 3, 4])) & (df["condition"].isin([1, 2, 3, 4, 5])) &
            (df["grade"].isin(range(1, 14))) & (df["sqft_above"] >= 300) & (df["sqft_above"] <= 9000) &
            (df["sqft_basement"] >= 0) & (df["sqft_basement"] <= 4800) & (df["yr_built"] >= 1900) &
            (df["yr_built"] <= 2015) & (df["sqft_living15"] >= 400) & (df["sqft_living15"] <= 6000)]

    return df


def fix_invalid_values(X):
    X.loc[~X["bedrooms"].isin(range(12)), "bedrooms"] = np.round(mean_feature_dict["bedrooms"])
    X.loc[~(X["bathrooms"] >= 0) & ~(X["bathrooms"] <= 8), "bathrooms"] = \
        np.round(mean_feature_dict["bathrooms"])
    X.loc[~(X["sqft_living"] >= 300) & ~(X["sqft_living"] <= 120000), "sqft_living"] = \
        np.round(mean_feature_dict["sqft_living"])
    X.loc[~(X["floors"] >= 1) & ~(X["floors"] <= 3.5), "floors"] = \
        np.round(mean_feature_dict["floors"])
    X.loc[~X["waterfront"].isin([0, 1]), "waterfront"] = np.round(mean_feature_dict["waterfront"])
    X.loc[~X["view"].isin([0, 1, 2, 3, 4]), "view"] = np.round(mean_feature_dict["view"])
    X.loc[~X["condition"].isin([1, 2, 3, 4, 5]), "condition"] = np.round(mean_feature_dict["condition"])
    X.loc[~X["grade"].isin(range(1, 14)), "grade"] = np.round(mean_feature_dict["grade"])
    X.loc[~(X["sqft_above"] >= 300) & ~(X["sqft_above"] <= 9000), "sqft_above"] = \
        np.round(mean_feature_dict["sqft_above"])
    X.loc[~(X["sqft_basement"] >= 0) & ~(X["sqft_basement"] <= 4800), "sqft_basement"] \
        = np.round(mean_feature_dict["sqft_basement"])
    X.loc[~(X["yr_built"] >= 1900) & ~(X["yr_built"] <= 2015), "yr_built"] = np.round(mean_feature_dict["yr_built"])
    X.loc[~(X["sqft_living15"] >= 400) & ~(X["sqft_living15"] <= 6000), "sqft_living15"] = \
        np.round(mean_feature_dict["sqft_living15"])

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X.loc[:, ~X.columns.str.startswith('zipcode_')]:
        pearson_correlation = (np.cov(X[feature], y)[1, 0]) / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(x=X[feature], y=y, trendline="ols")
        fig.update_layout(title=f"Correlation Between {feature} (feature) and price (response). "
                                f" Pearson Correlation is: {pearson_correlation}",
                          xaxis_title=f"{feature}",
                          yaxis_title="price")
        fig.write_image(output_path + f"/PearsonCorrelationOf{feature}.png", width=1500, height=1000)


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    X = df.drop(["price"], axis=1)
    y = df["price"]
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 2 - Preprocessing of housing prices dataset
    processed_train_x, processed_train_y = preprocess_data(train_x, train_y)  # TODO
    temp_test_y = test_y
    test_y.dropna(inplace=True)
    invalid_rows_of_test_price = temp_test_y.index.difference(test_y.index)
    without_zipcode = processed_train_x.loc[:, ~processed_train_x.columns.str.startswith('zipcode_')]
    mean_feature_dict = dict(without_zipcode.mean(axis=0))
    processed_test_x = preprocess_data(test_x)
    processed_test_x = processed_test_x.reindex(columns=processed_train_x.columns, fill_value=0)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(processed_train_x, processed_train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_loss_mat = np.zeros((91, 10))
    for row, p in enumerate(range(10, 101)):
        fraction = p / 100
        for col in range(10):
            sampled_train_data = processed_train_x.sample(frac=fraction)
            sampled_response = processed_train_y.loc[sampled_train_data.index]
            linear_reg = LinearRegression(include_intercept=True)
            linear_reg._fit(sampled_train_data, sampled_response)
            mean_loss_mat[row, col] = linear_reg._loss(processed_test_x.to_numpy(), test_y)
    mean_pred, std_pred = mean_loss_mat.mean(axis=1), mean_loss_mat.std(axis=1)
    fig = go.Figure((go.Scatter(x=list(range(10, 101)), y=mean_pred, mode="markers+lines",
                                line=dict(color="black"), marker=dict(color="black"), showlegend=False),
                     go.Scatter(x=list(range(10, 101)), y=mean_pred - 2 * std_pred, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=list(range(10, 101)), y=mean_pred + 2 * std_pred, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"), showlegend=False)))
    fig.update_layout(title="MSE of Test Data as Function of Increasing Training Data",
                      xaxis_title="Training Data Percentage",
                      yaxis_title="MSE Of Test Data")
    fig.show()
