# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:33:25 2019

@author: Santosh Sah
"""
import nltk
from MDIProjectSentimentAnalysis import (mdiProjectExtractFeatures, mdiProjectGetFeatureVectorForSingleTweet, mdiProjectGetFeatureListAndTweetListForTrainingSet, 
                                         mdiProjectProcessTweet, mdiProjectReplaceTwoOrMore, mdiProjectStopWordList)

testTweet = "Old Quarter is the best place to visit in Hanoi.. excellent and awesome ambience"

mdiProjectStopWordsList = mdiProjectStopWordList("files/StopWords.txt")
mdiProcessTweet = mdiProjectProcessTweet(testTweet)


mdiProjectGetFeatureVectorList = mdiProjectGetFeatureVectorForSingleTweet(mdiProcessTweet,mdiProjectStopWordsList)

feature_words = mdiProjectExtractFeatures(mdiProjectGetFeatureVectorList)