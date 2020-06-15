#CSE 512 HW 1 readme.txt

Implements the perceptron and AdaBoost (using decision stumps as weak learners) algorithms from scratch. Also implemented k-fold cross validation for use in verifying efficacy of both algorithms.

perceptron.py usage:
python perceptron.py --dataset /path/to/data/filename.csv --mode erm
python perceptron.py --dataset linearly-separable-dataset.csv --mode cv
mode can be erm (empirical risk minimization) or cv (10-fold cross validation), dataset must be a path to the dataset in question


adaboost.py usage:
python adaboost.py --dataset Breast_cancer_data.csv --mode cv --rounds 4
python adaboost.py --dataset Breast_cancer_data.csv --mode erm --rounds 4
please include an integer rounds parameter for the number of boosting rounds.
