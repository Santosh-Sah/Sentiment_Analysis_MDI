# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:04:39 2019

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle 

from MDIProjectSentimentAnalysis import (mdiProjectExtractFeatures, mdiProjectGetFeatureVectorForSingleTweet, 
                                         mdiProjectProcessTweet, mdiProjectStopWordList, mdiProjectReadFiles, mdiProjectReadSampleFile,
                                         mdiProjectSplitTrainTest)
def naiveBayesClassifierTestModel():
    
    mdiProjectSampleTweetList = mdiProjectReadSampleFile(mdiProjectReadFiles("mdiProjectFiles","SampleTrainingData.csv"))
    mdiProjectTrainingSet, mdiProjectTestingSet = mdiProjectSplitTrainTest(mdiProjectSampleTweetList)
    mdiProjectStopWordsList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles","StopWords.txt"))
    
    #mdiProjectTestingSet = mdiProjectTestingSet[1:10]
    
    mdiProjectActualSentiments = []
    mdiProjectPredictedSentiments = []
    
    #Open NaiveBayesClassifier picke file
    with open("NaiveBayesClassifierModel.pkl","rb") as NaiveBayesClassifier:
        NaiveBayesClassifierModel = pickle.load(NaiveBayesClassifier)
        
    for mdiProjectTestingTweetRow in mdiProjectTestingSet:
        
        mdiProjectSentiment = mdiProjectTestingTweetRow[0]
        mdiProjectTweet = mdiProjectTestingTweetRow[1]
    
        #Process the tweet
        porcessedTweet = mdiProjectProcessTweet(mdiProjectTweet)
        
        #Get the features vector for a single tweet
        featureVector = mdiProjectGetFeatureVectorForSingleTweet(porcessedTweet, mdiProjectStopWordsList)
        
        #Get the feacture words
        featureWords = mdiProjectExtractFeatures(featureVector)
        
        #Get sentiments based on the feature words
        tweetSentiment = NaiveBayesClassifierModel.classify(featureWords)
        
        mdiProjectActualSentiments.append(mdiProjectSentiment)
        mdiProjectPredictedSentiments.append(tweetSentiment)
    
    return mdiProjectActualSentiments, mdiProjectPredictedSentiments

def naiveBayesClassifierTestModelResult():
    
    mdiProjectActualSentiments, mdiProjectPredictedSentiments = naiveBayesClassifierTestModel()
    #print(mdiProjectActualSentiments)
    #print(mdiProjectPredictedSentiments)

    for actualSentimentPos, actualSentiment in enumerate(mdiProjectActualSentiments):
        
        if(actualSentiment == "positive"):
            
            mdiProjectActualSentiments[actualSentimentPos] = 1
        else:
            mdiProjectActualSentiments[actualSentimentPos] = 0
    
    for predictedSentimentPos, predictedSentiment in enumerate(mdiProjectPredictedSentiments):
        
        if(predictedSentiment == "positive"):
            
            mdiProjectPredictedSentiments[predictedSentimentPos] = 1
        else:
            mdiProjectPredictedSentiments[predictedSentimentPos] = 0
            
    result = confusion_matrix(mdiProjectActualSentiments, mdiProjectPredictedSentiments)
    print(result)
    
    accuracy = accuracy_score(mdiProjectActualSentiments, mdiProjectPredictedSentiments)
    print(accuracy)
    
    print(classification_report(mdiProjectActualSentiments, mdiProjectPredictedSentiments))
    
def logisticRegressionClassifierTestModel():
    
    #Load logistic regression classification pickle file
    with open("mdiProjectLogisticRegressionclassifier.pkl","rb") as mdiProjectLogisticRegressionclassifier:
        mdiProjectLogisticRegressionclassifierModel = pickle.load(mdiProjectLogisticRegressionclassifier)
    
    #Load X_test data set pickle file
    with open("mdiProjectLogisticRegressionX_test.pkl","rb") as mdiProjectLogisticRegressionX_test:
        X_test = pickle.load(mdiProjectLogisticRegressionX_test)
    
    #Load y_test data set pickle file
    with open("mdiProjectLogisticRegressionY_test.pkl","rb") as mdiProjectLogisticRegressionY_test:
        y_test = pickle.load(mdiProjectLogisticRegressionY_test)
    
    #Load X_train data set pickle file
    with open("mdiProjectLogisticRegressionX_train.pkl","rb") as mdiProjectLogisticRegressionX_train:
        X_train = pickle.load(mdiProjectLogisticRegressionX_train)
    
    #Load y_train data set pickle file
    with open("mdiProjectLogisticRegressionY_train.pkl","rb") as mdiProjectLogisticRegressionY_train:
        y_train = pickle.load(mdiProjectLogisticRegressionY_train)
        
    #Predict for X_train data
    y_train_predict = mdiProjectLogisticRegressionclassifierModel.predict(X_train)
    
    #Predict for X_test data
    y_test_predict = mdiProjectLogisticRegressionclassifierModel.predict(X_test)
    
    #Confussion matrix for training set
    confusionMatrixTrain = confusion_matrix(y_train, y_train_predict)
    
    #Confussion matrix for test set
    confusionMatrixTest = confusion_matrix(y_test, y_test_predict)
    
    print(confusionMatrixTrain)
    print(confusionMatrixTest)
    
    #Classification report of test set
    print(classification_report(y_test,y_test_predict))
    
    #Classification report of training set
    print(classification_report(y_train,y_train_predict))
    
    #Model accuracy for test data
    testAccuracy = accuracy_score(y_test, y_test_predict)
    
    #Model accuracy for training data
    trainAccuracy = accuracy_score(y_train, y_train_predict)
    
    print(testAccuracy)
    print(trainAccuracy)
    
if __name__ == "__main__":
    logisticRegressionClassifierTestModel()
    