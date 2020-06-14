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




def adaBoost(df,rounds=7):
	'''Takes in a dataframe df and optional rounds parameter (default 10)
	and performs the adaboost learning algorithm.'''

	#Represent success and failure as -1 and 1
	df.iloc[:,-1] = df.iloc[:,-1].map(lambda y: -1.0 if y == 0 else 1.0)
	df["weight"] = 1.0/len(df)
	learners = []
	for i in range(rounds):
		#invoke weak learner
		jstar, Thetastar = decisionStump(df,i)
		learners.append((jstar,Thetastar))
		epsilon = sum(df[((df.iloc[:,jstar]>Thetastar) & (df.iloc[:,-(2+i)]==1)) | ((df.iloc[:,jstar]<=Thetastar) & (df.iloc[:,-(2+i)]==-1))].iloc[:,-1])
		w = 0.5*np.log((1.0/epsilon)-1.0)
		df["stump"] =  df.iloc[:,jstar].map(lambda x: -1.0 if x>Thetastar else 1.0)
		denomSum = sum(df.apply(lambda x: x.iloc[-(2+i)] * math.exp(-w*x.iloc[-(3+i)]*x.iloc[-1]),axis=1))
		weightName = "weight" + str(i)
		df[weightName] = df.apply(lambda x: x.iloc[-(2+i)] * math.exp(-w*x.iloc[-(3+i)]*x.iloc[-1]) / denomSum,axis=1)
		df = df.drop("stump",axis=1)
	df = df.drop(weightName,axis=1)

	#Compute number of failures using strong learner
	df["sum"] = 0
	for j in range(rounds):
		jstar = learners[j][0]
		Thetastar = learners[j][1]
		df["stump"] = df.iloc[:,jstar].map(lambda x: -1.0 if x>Thetastar else 1.0)
		df["sum"] = df["sum"] + (df["stump"]* df.iloc[:,-(3+(rounds-1-j))])
		df = df.drop("stump",axis=1)
	df["hypothesis"] = df["sum"].map(lambda x: -1.0 if x<=0 else 1.0)
	print(len((df[df["hypothesis"]!= df["diagnosis"]])))



def kFoldCrossValidation(df,algorithm=adaBoost,k=10):
	'''Takes in a dataframe df, and optional algorithm (default adaboost),
	and k-value (default 10). Returns the weight vector with the lowest error
	on the validation data.'''
	pass

def empiricalRiskMin(df,algorithm=adaBoost):
	'''Takes in a dataframe df and optional algorithm (default adaboost).
	Returns computed weight vector using training data.'''
	pass

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
args = parser.parse_args()

df = pd.read_csv(args.dataset)
adaBoost(df)

# if args.mode == "cv":
# 	kFoldCrossValidation(df,adaBoost)
# elif args.mode == "erm":
# 	empiricalRiskMin(df)