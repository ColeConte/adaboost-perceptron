Cole Conte CSE 512 HW 1
readme.txt

4. README Section
perceptron.py usage:
python perceptron.py --dataset /path/to/data/filename.csv --mode erm
mode can be erm or cv

python perceptron.py --dataset linearly-separable-dataset.csv --mode cv

adaboost.py follows the same pattern



1.
2. I observed that the perceptron algorithm did not terminate for the non-linearly separable data set. The algorithm will continually update its weights until all the points are separated, but because not all points are linearly separable, the algorithm does not reach its termination condition and will just continue updating. It's a robust algorithm as long as the realizability assumption holds.

To make the algorithm terminate when the realizability assumption does not hold, I first researched upper bounds on the convergence of the perceptron algorithm. The proven upper bound, however, appears to be a factor of the maximum norm of an input vector, and since we haven't specified constraints on the input vector, we can't use it. Unable to find any additional guidance online, I decided to set the maximum number of iterations as the size of the data set. This makes 10-fold cross validation run rather slowly.


3.