import numpy as np
import matplotlib.pyplot as plt
import os
import math
from numpy.linalg import inv as inv_

from algorithms import Algorithms

plt.rcParams["figure.figsize"] = (20, 20)
plt.style.use("ggplot")


def read_data(file_name):
    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    all_data = []
    for data in lines:
        temp_data = []
        for vals in data.split(","):
            temp_data.append(float(vals))
        all_data.append(temp_data)
    return all_data


iris = read_data("pp3data\irlstest.csv")


def read_labels(file_name):
    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    lines = [int(line) for line in lines]
    return lines

iris_labels = read_labels("pp3data\labels-irlstest.csv")

bayesian_algo_iris = Algorithms(iris,iris_labels)
error_rate, weight_matrix, time_counter, errors_weight = bayesian_algo_iris.BayesianLogisticRegression(logger=True,shuffle=False)

print("The final convergence weights are: ")
print(weight_matrix[-1])