# COMPARISON BETWEEN GENERATIVE MODEL AND BAYESIAN LOGISTIC REGRESSION USING BOTH NEWTON-RAPHSON METHOD AND GRADIENT DESCENT

## The repository has two tasks, (I)It compares the generative model with the Bayesian Model (Newton update) (II)And, it compares the update quality for both Newton update and Gradient descent

### There are three important files here `algorithms.py`, which has the bayesian and vanilla linear regression algorithm. `task_1.py` and `task_2.py` are the plotting and implementation files for Task 1 and Task 2 respectively.

Run this for Task 1 
```
python task_1.py
```
### Outputs:
It will output three plots, `A.png`, `B.png` and `USPS.png`. Each plot has the trend for the error rate for both generative and bayesian discriminative model, for all three datasets.

#### Run this for Task 2 
```
python task_2.py
```
### Outputs:
#### It outputs 2 plots: `A_T2.jpg` and `USPS_T2.jpg` for both datasets and it has the plot Error rate vs. Time taken for update till convergence.

### Run the iris dataset to check the weights
```
python test_iris.py
```
