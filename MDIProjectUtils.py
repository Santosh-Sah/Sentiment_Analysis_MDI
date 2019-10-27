# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:47:51 2019

@author: Santosh Sah
"""

from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import pandas as pd
from MDIProjectSentimentAnalysis import (mdiProjectReadFiles, mdiProjectReadSampleFile, mdiProjectStopWordList)

mdiProjectStopWordsList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles","StopWords.txt"))
mdiProjectPunctuationList = set(string.punctuation)
mdiProjectLemmitization = WordNetLemmatizer()
mdiProjectCleanedTweetList = []

#Clean tweet list
def mdiProjectCleanTweet(mdiProjectTweetList):
    
    #Looping over list of tweets
    for i in range(len(mdiProjectTweetList)):
        
        mdiProjectTweet = re.sub('[^a-zA-Z]', ' ', mdiProjectTweetList[i])
        
        #Lowering the case of tweet
        mdiProjectTweet = mdiProjectTweet.lower()
        
        #Split the tweet
        mdiProjectTweet = mdiProjectTweet.split()
        
        #Lemmitize over tweet
        mdiProjectTweet = [mdiProjectLemmitization.lemmatize(word) for word in mdiProjectTweet if not word in mdiProjectStopWordsList]
        mdiProjectTweet = ' '.join(mdiProjectTweet)
        
        mdiProjectCleanedTweetList.append(mdiProjectTweet)
        
    return mdiProjectCleanedTweetList

def mdiProjectProcessSampleFileToTrainModel():
    
    #Read sample tweets
    mdiProjectSampleTweetList = mdiProjectReadSampleFile(mdiProjectReadFiles("mdiProjectFiles","SampleTrainingData.csv"))
    mdiProjectSampleTweetList = list(mdiProjectSampleTweetList)
    mdiProjectSampleTweetList = mdiProjectSampleTweetList[1:len(mdiProjectSampleTweetList)]
    
    #Create data frame out of the sample file tweets
    mdiProjectDataFrame = pd.DataFrame(mdiProjectSampleTweetList, columns=['sentiments','text'])

    mdiProjectTweetTextList = mdiProjectDataFrame['text']
    mdiProjectTweetSentimentsList = mdiProjectDataFrame['sentiments']
    
    #Clean tweet list
    mdiProjectTweetTextListClean = mdiProjectCleanTweet(mdiProjectTweetTextList)
    
    mdiProjectCleanedDataFrame = pd.DataFrame({'text': mdiProjectTweetTextListClean,'sentiments':mdiProjectTweetSentimentsList})
    
    return mdiProjectCleanedDataFrame
    
if __name__ == "__main__":
    mdiProjectProcessSampleFileToTrainModel()

    
    