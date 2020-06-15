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

	df.iloc[:,-1] = df.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
	df["bias"] = 1.0
	cols = df.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	df = df[cols]
	w = pd.DataFrame(np.zeros((1,len(df.columns)-1)))
	w.columns = df.columns.values[:-1]
	for _ in range(len(df)):
		i=0
		while(df.iloc[i,:-1].dot(w.iloc[0])*df.iloc[i,-1] >0):
			i+=1
			if(i==len(df)):
				return w
		w = pd.DataFrame(w + (df.iloc[i,-1]*df.iloc[i,:-1]))	
	return w


def kFoldCrossValidation(df,algorithm=perceptron,k=10):
	'''Takes in a dataframe df, and optional algorithm (default perceptron),
	and k-value (default 10). Returns the weight vector with the lowest error
	on the validation data.'''

	#Split train/test
	trainAndValidation,test = trainTestSplit(df)

	#Determine folds for cross-validation
	print("Splitting training data into "+str(k)+" folds for cross-validation.\n")
	trainAndValidation["fold"] = [random.randint(0,k-1) for i in range(len(trainAndValidation))]
	errors = {}
	sumErrors = 0
	minError = 1
	minErrorPredictor = None

	#Perform cross-validation to pick the best hyperparameters
	for i in range(k):
		validation = trainAndValidation[trainAndValidation["fold"]==i]
		train = trainAndValidation[trainAndValidation["fold"]!=i]
		validation = validation.drop(labels="fold",axis=1)
		train = train.drop(labels="fold",axis=1)

		#Represent success and failure as -1 and 1
		validation.iloc[:,-1] = validation.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
		
		#Use training set to train on algorithm
		output = algorithm(train.iloc[:,:])

		#Test for errors on validation set
		validation["bias"] = 1.0
		cols = validation.columns.tolist()
		cols = cols[-1:] + cols[:-1]
		validation = validation[cols]

		for j in range(len(validation)):
			if(validation.iloc[j,:-1].dot(output.iloc[0])*validation.iloc[j,-1] <= 0):
				if i in errors:
					errors[i]["count"]+=1
				else:
					errors[i] = {"count":1,"pct":0}
			if i not in errors:
				errors[i] = {"count":0,"pct":0}
		errors[i]["pct"] = errors[i]["count"]/float(len(validation))
		sumErrors += errors[i]["pct"]
		if(errors[i]["pct"]<minError):
			minErrorPredictor = output
			minError = errors[i]["pct"]
		print("Fold number "+str(i)+" had an error rate of: "+str(errors[i]["pct"])+\
			"% \nWith a weight vector:\n" + str(output))

	#Final results printout
	print("Mean error averaged across all folds was: " +str(100*sumErrors/k)+"%")
	print("Predictor with lowest error is selected. Weight vector is:\n"+ str(minErrorPredictor))

	#Compute error on test dataset
	testErrors = 0
	test["bias"] = 1.0
	cols = test.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	test = test[cols]

	for j in range(len(test)):
		if(test.iloc[j,:-1].dot(minErrorPredictor.iloc[0])*test.iloc[j,-1] <= 0):
			testErrors+=1
	print("Error on test data is: "+str(100*testErrors/float(len(test)))+"%")
	return minErrorPredictor

def empiricalRiskMin(df,algorithm=perceptron):
	'''Takes in a dataframe df and optional algorithm (default perceptron).
	Returns computed weight vector using training data.'''

	#Split train/test
	train,test = trainTestSplit(df)

	#Represent success and failure as -1 and 1
	train.iloc[:,-1] = train.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
	test.iloc[:,-1] = test.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
		
	#Use training set to train on algorithm
	output = algorithm(train.iloc[:,:])

	#Get empirical risk
	train["bias"] = 1.0
	cols = train.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	train = train[cols]
	trainErrors = 0
	for j in range(len(train)):
		if(train.iloc[j,:-1].dot(output.iloc[0])*train.iloc[j,-1] <= 0):
			trainErrors+=1

	print("Error on train data is: "+str(100*trainErrors/float(len(train)))+"%")
	print("using weight vector:\n" + str(output))

	#Test for errors on test set
	test["bias"] = 1.0
	cols = test.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	test = test[cols]
	testErrors = 0
	for j in range(len(test)):
		if(test.iloc[j,:-1].dot(output.iloc[0])*test.iloc[j,-1] <= 0):
			testErrors+=1

	print("Error on test data is: "+str(100*testErrors/float(len(test)))+"%")


def trainTestSplit(df):
	random.seed(13579)
	print("Splitting data into 80% training, 20% testing.\n")
	df["trainTest"] = [random.random() for i in range(len(df))]
	test = df[df["trainTest"]>=0.8]
	test = test.drop(labels="trainTest",axis=1)
	train = df[df["trainTest"]<0.8]
	train = train.drop(labels="trainTest",axis=1)
	return train,test


parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--mode")
args = parser.parse_args()

df = pd.read_csv(args.dataset)
if args.mode == "cv":
	kFoldCrossValidation(df,perceptron)
elif args.mode == "erm":
	empiricalRiskMin(df)





