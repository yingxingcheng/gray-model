#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def calc_params(X0):
    """
    Calculate the parameters a and b for the GM(1,1) model.

    Parameters
    ----------
    X0 : np.array
        The input time series data.

    Returns
    -------
    a, b : tuple
        The parameters a and b for the GM(1,1) model.
    """
    X1 = np.cumsum(X0)
    B = np.ones((X0.size - 1, 2))
    B[:, 0] = -(X1[:-1] + X1[1:]) / 2
    return np.linalg.inv(B.T @ B) @ B.T @ X0[1:]


def calc_X1_hat(X0, a, b, k):
    """
    Calculate the estimated value of X1.

    Parameters
    ----------
    X0 : np.array
        The input time series data.
    a, b : float
        The parameters a and b for the GM(1,1) model.
    k : int
        The index of the data point.

    Returns
    -------
    X1_hat : float
        The estimated value of X1.
    """
    return (X0[0] - b / a) * np.exp(-a * k) + b / a


def predict(X0, a, b, nb_k):
    """
    Predict the future values of the time series.

    Parameters
    ----------
    X0 : np.array
        The input time series data.
    a, b : float
        The parameters a and b for the GM(1,1) model.
    nb_k : int
        The number of steps to predict.

    Returns
    -------
    X0_hat : np.array
        The predicted time series.
    """
    k = np.arange(0, nb_k)
    X1_hat = calc_X1_hat(X0, a, b, k)
    X0_hat = np.zeros((nb_k,))
    X0_hat[0] = X0[0]
    X0_hat[1:] = X1_hat[1:] - X1_hat[:-1]
    return X0_hat


def do_stat(X0, X0_hat):
    """
    Print the relative error.

    Parameters
    ----------
    X0 : np.array
        The original time series data.
    X0_hat : np.array
        The predicted time series data.
    """
    X0_mean = np.mean(X0)
    S1_sq = np.var(X0)
    S2_sq = np.var(X0 - X0_hat)
    C = np.sqrt(S2_sq) / np.sqrt(S1_sq)
    print(f"C = {C}")


def make_plot(X0, X0_hat):
    """
    Plot the original and predicted time series.

    Parameters
    ----------
    X0 : np.array
        The original time series data.
    X0_hat : np.array
        The predicted time series data.
    """
    fig, ax = plt.subplots()
    ax.plot(range(X0_hat.size), X0_hat, color="tab:blue")
    ax.scatter(range(X0.size), X0, color="tab:red")
    plt.show()


def main():
    # DOI:10.4236/gep.2017.59011
    X0 = np.array([32.300, 48.600, 69.600, 96.370, 128.700, 168.200, 207.870, 256.400, 306.800])
    for idx in range(3):
        print("#" * 80)
        print(f"Model {idx+1}")
        X0_used = X0[idx : 6 + idx]
        a, b = calc_params(X0_used)
        print(f"parameters a = {a} and b = {b}")
        X0_hat = predict(X0_used, a, b, len(X0) - idx)
        print("X0_hat:", X0_hat)
        print("X0:", X0)
        do_stat(X0[idx:], X0_hat)
        # make_plot(X0[idx:], X0_hat)


if __name__ == "__main__":
    main()
