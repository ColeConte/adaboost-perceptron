'''
Cole Conte CSE 512 HW 1
adaboost.py
'''

import argparse
import math
import pandas as pd
import numpy as np
import random
from itertools import count, chain


def decisionStump(df,i=0):
	'''Takes in all values of a weighted dataframe and the round of learning (to avoid saved weight vectors)
	and returns an optimal decision stump.'''
	Fstar = float("inf")
	Thetastar = None
	jstar = None
	for j in range(len(df.columns)-(2+i)):
		sortedDf = df.sort_values(by=df.columns.values[j])
		F = sum(sortedDf[sortedDf.iloc[:,-(2+i)]==1].iloc[:,-1])
		if F < Fstar:
			Fstar = F
			Thetastar = sortedDf.iloc[1,j] -1
			jstar = j
		for k in range(len(sortedDf)):
			F = F - sortedDf.iloc[k,-(2+i)]*sortedDf.iloc[k,-1]
			if F < Fstar and (k!=len(sortedDf)-1) and (sortedDf.iloc[k,j] != sortedDf.iloc[k+1,j]):
				Fstar = F
				Thetastar = .5 *(sortedDf.iloc[k,j] + sortedDf.iloc[k+1,j])
				jstar = j
	return jstar, Thetastar




def adaBoost(df,rounds):
	'''Takes in a dataframe df and optional rounds parameter (default 10)
	and performs the adaboost learning algorithm.'''

	#Represent success and failure as -1 and 1
	df["weight"] = 1.0/len(df)
	learners = []
	print("Performing "+str(rounds) +" rounds of boosting.")
	for i in range(rounds):
		#invoke weak learner
		jstar, Thetastar = decisionStump(df,i)
		epsilon = sum(df[((df.iloc[:,jstar]>Thetastar) & (df.iloc[:,-(2+i)]==1)) | ((df.iloc[:,jstar]<=Thetastar) & (df.iloc[:,-(2+i)]==-1))].iloc[:,-1])
		w = 0.5*np.log((1.0/epsilon)-1.0)
		learners.append([jstar,Thetastar,w])
		df["stump"] =  df.iloc[:,jstar].map(lambda x: -1.0 if x>Thetastar else 1.0)
		denomSum = sum(df.apply(lambda x: x.iloc[-(2+i)] * math.exp(-w*x.iloc[-(3+i)]*x.iloc[-1]),axis=1))
		weightName = "weight" + str(i)
		df[weightName] = df.apply(lambda x: (x.iloc[-(2+i)] * math.exp(-w*x.iloc[-(3+i)]*x.iloc[-1])) / denomSum,axis=1)
		df = df.drop("stump",axis=1)
	df = df.drop(weightName,axis=1)

	return learners
	

def kFoldCrossValidation(df,algorithm=adaBoost,k=10):
	'''Takes in a dataframe df, and optional algorithm (default adaboost),
	and k-value (default 10). Returns the weight vector with the lowest error
	on the validation data.'''
	pass

def empiricalRiskMin(df,algorithm=adaBoost,trainingRounds=10):
	'''Takes in a dataframe df and optional algorithm (default adaboost).
	Returns computed weight vector using training data.'''

	#Split train/test
	train,test = trainTestSplit(df)

	#Represent success and failure as -1 and 1
	train.iloc[:,-1] = train.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
	test.iloc[:,-1] = test.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)

	#Use training set to train on algorithm
	learners = algorithm(train.copy(),trainingRounds)
	train["sum"] = 0

	#Compute number of failures on train set using strong learner
	print("Strong learner consists of the following (jstar, Thetastar, w) stumps:")
	print(learners)

	for j in range(trainingRounds):
		jstar = learners[j][0]
		Thetastar = learners[j][1]
		w = learners[j][2]
		train["stump"] = train.iloc[:,jstar].map(lambda x: -1.0 if x > Thetastar else 1.0)
		train["sum"] = train["sum"] + (train["stump"]*w)
		train = train.drop("stump",axis=1)
	train["hypothesis"] = train["sum"].map(lambda x: -1.0 if x<=0 else 1.0)
	trainErrors = len((train[train["hypothesis"]!= train.iloc[:,-3]]))
	print("Error on train data is: "+str(100*trainErrors/float(len(train)))+"%")

	#Compute number of failures on test set using strong learner
	test["sum"] = 0
	for j in range(trainingRounds):
		jstar = learners[j][0]
		Thetastar = learners[j][1]
		w = learners[j][2]
		test["stump"] = test.iloc[:,jstar].map(lambda x: -1.0 if x>Thetastar else 1.0)
		test["sum"] = test["sum"] + (test["stump"]* w)
		test = test.drop("stump",axis=1)
	test["hypothesis"] = test["sum"].map(lambda x: -1.0 if x<=0 else 1.0)
	testErrors = len((test[test["hypothesis"]!= test.iloc[:,-3]]))
	print("Error on test data is: "+str(100*testErrors/float(len(test)))+"%")


def trainTestSplit(df):
	'''Takes in a dataframe and returns testing and training dataframes'''
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
parser.add_argument("--rounds")
args = parser.parse_args()

df = pd.read_csv(args.dataset)

if args.mode == "cv":
	kFoldCrossValidation(df,adaBoost,int(args.rounds))
elif args.mode == "erm":
	empiricalRiskMin(df, adaBoost, int(args.rounds))