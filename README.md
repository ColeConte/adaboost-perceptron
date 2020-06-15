# Perceptron and AdaBoost
Implemented the perceptron and AdaBoost (using decision stumps as weak learners) algorithms from scratch. Also implemented k-fold cross validation for use in verifying efficacy of both algorithms.

## Usage
perceptron.py usage:
```bash
python perceptron.py --dataset /path/to/data/filename.csv --mode mode
```
mode parameter can be erm (empirical risk minimization) or cv (10-fold cross validation), dataset must be a path to the dataset in question.

Example: 
```bash
python perceptron.py --dataset linearly-separable-dataset.csv --mode cv
```

adaboost.py example:
```bash
python adaboost.py --dataset Breast_cancer_data.csv --mode cv --rounds 4
```
mode parameter can be erm (empirical risk minimization) or cv (10-fold cross validation), dataset must be a path to the dataset in question. Please include an integer rounds parameter for the number of boosting rounds.
