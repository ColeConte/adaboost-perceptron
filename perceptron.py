'''
Cole Conte CSE 512 HW 1
perceptron.py
'''

import argparse
import pandas as pd
import numpy as np
from itertools import count

def perceptron(df):
	w = pd.DataFrame(np.zeros((1,len(df.columns)-1)))
	x = df.iloc[:,:-1]
	y = df.iloc[:,-1]
	y = y.replace(0,-1)
	w.columns = x.columns.values
	for t in count():
		for i in range(len(df)):
			if (x.iloc[i].dot(w.iloc[0]))*y.iloc[i] <= 0:
				w = pd.DataFrame(w.iloc[0] + (y.iloc[i]*x.iloc[i]))
				w = w.T
				print(i)
				break
		if i == (len(df)-1):
			return w

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--mode")
args = parser.parse_args()

df = pd.read_csv(args.dataset)
perceptron(df)


