Cole Conte CSE 512 HW 1
readme.txt

4. README Section
perceptron.py usage:
python perceptron.py --dataset /path/to/data/filename.csv --mode erm
python perceptron.py --dataset linearly-separable-dataset.csv --mode cv
mode can be erm (empirical risk minimization) or cv (10-fold cross validation), dataset must be a path to the dataset in question


adaboost.py usage:
python adaboost.py --dataset Breast_cancer_data.csv --mode cv --rounds 4
python adaboost.py --dataset Breast_cancer_data.csv --mode erm --rounds 4
please include an integer rounds parameter for the number of boosting rounds.


1.
Perceptron ERM: using the train/test split created by my seed,

Perceptron k-folds:

Adaboost ERM: using the train/test split created by my seed, I hit a cycle after two rounds of boosting (these stumps kept appearing).
[jstar, Thetastar, w] = [3, 696.25, 1.0500304144412858], [1, 18.634999999999998, 0.60051364704808541]
Error on train data is: 10.9090909091%
Error on test data is: 9.3023255814%

Adaboost k-folds: 

2. I observed that the perceptron algorithm did not terminate for the non-linearly separable data set. The algorithm will continually update its weights until all the points are separated, but because not all points are linearly separable, the algorithm does not reach its termination condition and will just continue updating. It's a robust algorithm as long as the realizability assumption holds.

To make the algorithm terminate when the realizability assumption does not hold, I first researched upper bounds on the convergence of the perceptron algorithm. The proven upper bound, however, appears to be a factor of the maximum norm of an input vector, and since we haven't specified constraints on the input vector, we can't use it. Unable to find any additional guidance online, I decided to set the maximum number of iterations as the size of the data set. This makes 10-fold cross validation run rather slowly.


3.