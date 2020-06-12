'''
Cole Conte CSE 512 HW 1
perceptron.py
'''

import argparse
import pandas as pd
import numpy as np
import random
from itertools import count, chain

def perceptron(df):
	'''Takes in a dataframe df and performs the perceptron learning algorithm. Returns
	the weight vector w.'''
	w = pd.DataFrame(np.zeros((1,len(df.columns)-1)))
	x = df.iloc[:,:-1]
	y = df.iloc[:,-1]
	y = y.replace(0,-1)
	w.columns = x.columns.values
	for t in count():
		for i in range(len(df)):
			if (x.iloc[i].dot(w.iloc[0]))*y.iloc[i] <= 0:
				w = pd.DataFrame(w.iloc[0]+ (y.iloc[i]*x.iloc[i]))
				w = w.T
				break
		if i == (len(df)-1):
			return w

def kFoldCrossValidation(df,algorithm=perceptron,k=10):
	'''Takes in a dataframe df, and optional algorithm (default perceptron),
	and k-value (default 10). Returns the weight vector with the lowest error.'''

	random.seed(13579)
	#Split into training and testing sets
	df["trainTest"] = [random.random() for i in range(len(df))]
	test = df[df["trainTest"]>=0.8]
	test = test.drop(labels="trainTest",axis=1)
	trainAndValidation = df[df["trainTest"]<0.8]
	trainAndValidation = trainAndValidation.drop(labels="trainTest",axis=1)

	#Determine folds for cross-validation
	trainAndValidation["fold"] = [random.randint(0,k-1) for i in range(len(trainAndValidation))]
	errors = {}
	sumErrors = 0
	minError = 1
	minErrorPredictor = None
	for i in range(k):
		validation = trainAndValidation[trainAndValidation["fold"]==i]
		train = trainAndValidation[trainAndValidation["fold"]!=i]
		validation = validation.drop(labels="fold",axis=1)
		train = train.drop(labels="fold",axis=1)

		#Represent success and failure as -1 and 1
		validation.iloc[:,-1] = validation.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
		
		#Sample perceptron output since its not converging rn 
		data = {"x1":[-20],
				"x2":[5]}
		output = pd.DataFrame(data,columns=["x1","x2"])
		#output = algorithm(train.iloc[:,:-1])

		#Test for errors
		for j in range(len(validation)):
			if(validation.iloc[j,:-1].dot(output.iloc[0])*validation.iloc[j,-1] <= 0):
				if i in errors:
					errors[i]["count"]+=1
				else:
					errors[i] = {"count":1,"pct":0}
		errors[i]["pct"] = errors[i]["count"]/float(len(validation))
		sumErrors += errors[i]["pct"]
		if(errors[i]["pct"]<minError):
			minErrorPredictor = output
		print("Fold number "+str(i)+" had an error rate of: "+str(errors[i]["pct"])+\
			"% \nWith a weight vector:\n" + str(output))

	#Final results printout
	print("Mean error averaged across all folds was: " +str(sumErrors/k)+"%")
	print("Final predictor selected is:\n"+ str(minErrorPredictor))
	return minErrorPredictor



parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--mode")
args = parser.parse_args()

df = pd.read_csv(args.dataset)
if args.mode == "cv":
	kFoldCrossValidation(df,perceptron)
else:
	perceptron(df)





