# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 09:40:04 2019

@author: Santosh Sah
"""
import nltk
import pickle 
from MDIProjectUtils import mdiProjectProcessSampleFileToTrainModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from MDIProjectSentimentAnalysis import (mdiProjectExtractFeatures, mdiProjectGetFeatureListAndTweetListForTrainingSet, mdiProjectReadFiles,
                                         mdiProjectReadSampleFile, mdiProjectSplitTrainTest, mdiProjectStopWordList)

def NaiveBayesClassifierModel():
    
    #Get sample tweets file
    mdiProjectSampleTweetList = mdiProjectReadSampleFile(mdiProjectReadFiles("mdiProjectFiles","SampleTrainingData.csv"))
    
    #Split into training and testing set
    mdiProjectTrainingSet, mdiProjectTestingSet = mdiProjectSplitTrainTest(mdiProjectSampleTweetList)
    
    #Get stop word list
    mdiProjectStopWordsList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles","StopWords.txt"))
    
    #Get tweets and feature list
    mdiProjectTweetsList, mdiProjectFeatureList = mdiProjectGetFeatureListAndTweetListForTrainingSet(mdiProjectTrainingSet, mdiProjectStopWordsList)
    
    #Get training set which has feature list
    training_set = nltk.classify.util.apply_features(mdiProjectExtractFeatures, mdiProjectTweetsList)
    
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    
    #Write naive bayes classifer as pickle file
    with open("NaiveBayesClassifierModel.pkl",'wb') as NaiveBayesClassifierModel_Pickle:
        pickle.dump(NBClassifier, NaiveBayesClassifierModel_Pickle, protocol = 2)

def LogisticsRegressionClassifierModel():
    
    #Get cleaned sample tweet list
    mdiProjectCleanedDataFrame = mdiProjectProcessSampleFileToTrainModel()
    
    #Get tweet sentiment list
    mdiProjectSentimentList = mdiProjectCleanedDataFrame['sentiments']
    
    #Get tweet text list
    mdiProjectTweetsList = mdiProjectCleanedDataFrame['text']
    
    # Creating the Tf-Idf model directly
    mdiProjectTfidfvectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6)
    X = mdiProjectTfidfvectorizer.fit_transform(mdiProjectTweetsList).toarray()
    y = mdiProjectSentimentList
    
    #Split train and test data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)
    
    print(X_train)
    
    #Get logistic regression classifier object
    mdiProjectLogisticRegressionclassifier = LogisticRegression()
    
    # Training the classifier
    mdiProjectLogisticRegressionclassifier.fit(X_train,y_train)
    
    #Write logistic regrssion classifer as pickle file
    with open("mdiProjectLogisticRegressionclassifier.pkl",'wb') as mdiProjectLogisticRegressionclassifier_Pickle:
        pickle.dump(mdiProjectLogisticRegressionclassifier, mdiProjectLogisticRegressionclassifier_Pickle, protocol = 2)
    
    #Write X_test as pickle file
    with open("mdiProjectLogisticRegressionX_test.pkl",'wb') as mdiProjectLogisticRegressionX_test_Pickle:
        pickle.dump(X_test, mdiProjectLogisticRegressionX_test_Pickle, protocol = 2)
    
    #Write y_test as pickle file
    with open("mdiProjectLogisticRegressionY_test.pkl",'wb') as mdiProjectLogisticRegressionY_test_Pickle:
        pickle.dump(y_test, mdiProjectLogisticRegressionY_test_Pickle, protocol = 2)
    
    #Write tfidfvectorizer as pickle file
    with open("mdiProjectTfidfvectorizer.pkl",'wb') as mdiProjectTfidfvectorizer_Pickle:
        pickle.dump(mdiProjectTfidfvectorizer, mdiProjectTfidfvectorizer_Pickle, protocol = 2)
    
    #Write X_train as pickle file
    with open("mdiProjectLogisticRegressionX_train.pkl",'wb') as mdiProjectLogisticRegressionX_train_Pickle:
        pickle.dump(X_train, mdiProjectLogisticRegressionX_train_Pickle, protocol = 2)
    
    #Write y_train as pickle file
    with open("mdiProjectLogisticRegressionY_train.pkl",'wb') as mdiProjectLogisticRegressionY_train_Pickle:
        pickle.dump(y_train, mdiProjectLogisticRegressionY_train_Pickle, protocol = 2)
    
if __name__ == "__main__":
    LogisticsRegressionClassifierModel() 
    