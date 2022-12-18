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


a = read_data("pp3data/A.csv")
b = read_data("pp3data/B.csv")
usps = read_data(r"pp3data\usps.csv")


def read_labels(file_name):
    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    lines = [int(line) for line in lines]
    return lines


a_labels = read_labels("pp3data/labels-A.csv")
b_labels = read_labels("pp3data/labels-B.csv")
usps_labels = read_labels("pp3data\labels-usps.csv")


def plot_error(gen_error: dict, blr_error: dict, title: str):
    mean_gen_dict = {key: sum(vals) / len(vals) for key, vals in gen_error.items()}
    std_gen_dict = {key: np.std(np.array(vals)) for key, vals in gen_error.items()}
    mean_blr_dict = {key: sum(vals) / len(vals) for key, vals in blr_error.items()}
    std_blr_dict = {key: np.std(np.array(vals)) for key, vals in blr_error.items()}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = mean_gen_dict.keys()
    ax.set_title(f"For dataset {title}")
    ax.errorbar(
        X, mean_gen_dict.values(), std_gen_dict.values(), label="Generative Model"
    )
    ax.errorbar(
        X, mean_blr_dict.values(), std_blr_dict.values(), label="Bayesian Model"
    )
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Fraction of Data Set Size")
    ax.legend(loc="best")
    fig.savefig(f"{title}.png")
    plt.show()


if __name__ == "__main__":
    data_fractions = [round(d, 1) for d in np.linspace(0.1, 1.0, 10)]
    a_algo = Algorithms(a, a_labels)
    b_algo = Algorithms(b, b_labels)
    usps_algo = Algorithms(usps, usps_labels)
    a_epochs_error_gen = {df: [] for df in data_fractions}
    a_epochs_error_blr = {df: [] for df in data_fractions}
    b_epochs_error_gen = {df: [] for df in data_fractions}
    b_epochs_error_blr = {df: [] for df in data_fractions}
    usps_epochs_error_gen = {df: [] for df in data_fractions}
    usps_epochs_error_blr = {df: [] for df in data_fractions}

    for epochs in range(1, 31):
        print("For Epoch")
        print(f"{epochs:-^20}")
        for data_fraction in data_fractions:
            a_gen_error = a_algo.generativeModel(data_fraction)
            b_gen_error = b_algo.generativeModel(data_fraction)
            usps_gen_error = usps_algo.generativeModel(data_fraction)
            a_epochs_error_gen[data_fraction].append(a_gen_error)
            b_epochs_error_gen[data_fraction].append(b_gen_error)
            usps_epochs_error_gen[data_fraction].append(usps_gen_error)
            a_blr_error, _, _, _ = a_algo.BayesianLogisticRegression(data_fraction)
            b_blr_error, _, _, _ = b_algo.BayesianLogisticRegression(data_fraction)
            usps_blr_error, _, _, _ = usps_algo.BayesianLogisticRegression(
                data_fraction
            )
            a_epochs_error_blr[data_fraction].append(a_blr_error)
            b_epochs_error_blr[data_fraction].append(b_blr_error)
            usps_epochs_error_blr[data_fraction].append(usps_blr_error)

    # print(list(a_epochs_error_gen.values()),list(a_epochs_error_blr.values()))
    plot_error(a_epochs_error_gen, a_epochs_error_blr, "A")
    plot_error(b_epochs_error_gen, b_epochs_error_blr, "B")
    plot_error(usps_epochs_error_gen, usps_epochs_error_blr, "USPS")
