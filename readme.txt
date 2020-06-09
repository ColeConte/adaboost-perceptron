Cole Conte CSE 512 HW 1
readme.txt

4. README Section
perceptron.py usage:
python perceptron.py --dataset /path/to/data/filename.csv --mode erm
mode can be erm or cv

adaboost.py follows the same pattern



1.
2. I observed that the perceptron algorithm did not terminate even after several minutes. This makes sense for a non-linearly separable data set; the algorithm will continually update its weights until all the points are separated, but because not all points are linearly separable, the algorithm will just cycle.
3.