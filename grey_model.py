#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def calc_params(x0):
    npoint = len(x0)
    x1 = np.cumsum(x0)

    B = np.ones((npoint - 1, 2))
    B[:, 0] = -(x1[:-1] + x1[1:]) / 2

    Y = x0[1:]
    a, b = np.linalg.inv(B.T @ B) @ B.T @ Y
    return a, b


def calc_x1_hat(x0, a, b, k):
    return (x0[0] - b / a) * np.exp(-a * k) + b / a


def predict(x0, a, b, nb_k):
    k = np.arange(0, nb_k + 0)
    x1_hat = calc_x1_hat(x0, a, b, k)
    x0_hat = np.zeros((nb_k,))
    x0_hat[0] = x0[0]
    x0_hat[1:] = x1_hat[1:] - x1_hat[:-1]
    return x0_hat


def do_stat(x0, x0_hat):
    x0_mean = np.mean(x0)
    S1_sq = np.var(x0)
    S2_sq = np.var(x0 - x0_hat)
    C = np.sqrt(S2_sq) / np.sqrt(S1_sq)
    print(C)


def make_plot(x0, x0_hat):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(npoint - 1), x0_hat, color="tab:blue")
    ax.scatter(range(npoint), x0, color="tab:red")
    plt.show()


if __name__ == "__main__":
    # DOI:10.4236/gep.2017.59011
    x0 = np.array([32.300, 48.600, 69.600, 96.370, 128.700, 168.200, 207.870, 256.400, 306.800])
    for idx in [0, 1, 2]:
        print("#" * 80)
        print(f"idx = {idx}")
        x0_used = x0[idx : 6 + idx]
        a, b = calc_params(x0_used)
        print("parameters a and b:")
        print(a, b)
        x0_hat = predict(x0_used, a, b, len(x0) - idx)
        print("x0_hat:")
        print(x0_hat)
        print("x0:")
        print(x0)
        do_stat(x0[idx:], x0_hat)
