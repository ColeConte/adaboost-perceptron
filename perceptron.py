'''
Cole Conte CSE 512 HW 1
perceptron.py
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--mode")
args = parser.parse_args()
print(args.dataset)


