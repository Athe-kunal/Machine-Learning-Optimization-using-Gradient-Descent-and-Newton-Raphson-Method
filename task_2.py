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
usps = read_data(r"pp3data\usps.csv")


def read_labels(file_name):
    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    lines = [int(line) for line in lines]
    return lines


a_labels = read_labels("pp3data/labels-A.csv")
usps_labels = read_labels("pp3data\labels-usps.csv")


def plot_errors(error,time,title:str,label:str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"For dataset {title}")
    ax.plot(time,error, marker='o',markersize=1,label=label)
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Time")
    ax.legend(loc="best")
    fig.savefig(f"{title}_{label}_T2.png")
    plt.show()

def plot_error_vs_time(blr_error, gd_error,blr_time,gd_time, title: str):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"For dataset {title}")
    ax.plot(blr_time,blr_error, marker='o',markersize=1,label="Newton Model")
    ax.plot(gd_time, gd_error, marker='o',markersize=1,label="GD Model")
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Time")
    ax.legend(loc="best")
    fig.savefig(f"{title}_T2.png")
    plt.show()


if __name__ == "__main__":
    a_algo = Algorithms(a, a_labels)
    usps_algo = Algorithms(usps, usps_labels)

    a_blr_time = []
    usps_blr_time = []
    a_gd_time = []
    usps_gd_time = []
    for epochs in range(1, 4):
        print(f"For epoch {epochs}")
        (
            _,
            a_weight_blr,
            a_time_blr,
            a_blr_error_weight,
        ) = a_algo.BayesianLogisticRegression(logger=True, shuffle=False)
        a_gd_error_weight, a_weight_gd, a_time_gd = a_algo.GradientDescentMethod(
            logger=True, shuffle=False
        )
        (
            _,
            usps_weight_blr,
            usps_time_blr,
            usps_blr_error_weight,
        ) = usps_algo.BayesianLogisticRegression(logger=True, shuffle=False)
        (
            usps_gd_error_weight,
            usps_weight_gd,
            usps_time_gd,
        ) = usps_algo.GradientDescentMethod(logger=True, shuffle=False)
        a_blr_time.append(a_time_blr)
        usps_blr_time.append(usps_time_blr)
        a_gd_time.append(a_time_gd)
        usps_gd_time.append(usps_time_gd)

    a_blr_time = list(np.mean(np.array(a_blr_time), axis=0))
    usps_blr_time = list(np.mean(np.array(usps_blr_time), axis=0))
    a_gd_time = list(np.mean(np.array(a_gd_time), axis=0))
    usps_gd_time = list(np.mean(np.array(usps_gd_time), axis=0))


    # print(a_blr_error_weight,a_gd_error_weight)

    plot_errors(a_blr_error_weight,a_blr_time,"A","Newton_Model")
    plot_errors(a_gd_error_weight,a_gd_time,"A","Gradient_Descent")
    plot_errors(usps_blr_error_weight,usps_blr_time,"USPS","Newton_Model")
    plot_errors(usps_gd_error_weight,usps_gd_time,"USPS","Gradient_Descent")
    
    plot_error_vs_time(a_blr_error_weight, a_gd_error_weight,a_blr_time,a_gd_time ,"A")
    plot_error_vs_time(usps_blr_error_weight, usps_gd_error_weight,usps_blr_time,usps_gd_time, "USPS")