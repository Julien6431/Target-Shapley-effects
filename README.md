# Target-Shapley-effects
Software and data related to the paper "Shapley effects estimation in reliability-oriented sensitivity analysis with dependent inputs by importance sampling".

## Repository structure

The folder "cross_entropy" contains the softwares related to the cross-entropy algorithms in order to estimate the failure probability by adaptive parametric importance sampling and to compute an adapted importance sampling auxiliary distribution.

The folder "shapley_estimators" contains the algorithms used the estimation of the target Shapley effects. The main functions can be found in the file "ROSA_shapley_effects.py". Estimators without and with importance sampling in both geven-model and given-data frameworks are implemented.

The folder "numerical_tests" contains the implementation of the test cases from the article as well as the corresponding data. These files illustrate how to use the algorithms. 

## Requirements

In order to install the required modules, please run the following line in a terminal or in the console:

```
pip install -r requirements.txt
```
