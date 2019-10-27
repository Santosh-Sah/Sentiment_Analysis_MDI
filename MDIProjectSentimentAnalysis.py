# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:19:29 2019

@author: Santosh Sah
"""
import csv
import re
import os
from sklearn.model_selection import train_test_split


#Process tweet
#Substitute  words like http, www etc with URL
#@username  words like http, www etc with AT_USER
#Remove white spaces, Substitute words with # with other words and trimming white spaces
def mdiProjectProcessTweet(mdiProjectTweet):
    
    #convert the tweet in lower case
    mdiProjectTweet = mdiProjectTweet.lower()
    
    #convert www.* or https?:// to URL
    mdiProjectTweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',mdiProjectTweet)
    
    #convert @username to AT_USER
    mdiProjectTweet = re.sub('@[^\s]+','AT_USER',mdiProjectTweet)   
    
    #Remove additional white spaces
    mdiProjectTweet = re.sub('[\s]+', ' ', mdiProjectTweet)
    
    #Replace #word with word
    mdiProjectTweet = re.sub(r'#([^\s]+)', r'\1', mdiProjectTweet)
                             
    #Trimming
    mdiProjectTweet = mdiProjectTweet.strip('\'"')
    
    return mdiProjectTweet

#If a tweet has more than one same word, replacing it with singlewords
def mdiProjectReplaceTwoOrMore(mdiProjectWord):
    
    #Look for 2 or more repetitions of the character and replace with the character itself
    mdiProjectPattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    
    return mdiProjectPattern.sub(r"\1\1", mdiProjectWord)

#In English language there are many stop words, which are not important in sentiment analysis.
#Project has a file which has a list of stop words. Read the file and populate a list of these stop words.
def mdiProjectStopWordList(mdiProjectStopWordListFileName):
    
    #Read stop words file and build a list
    mdiProjectStopWordList = []
    
    #Populating stopword list with AT_USER, URL
    mdiProjectStopWordList.append('AT_USER')
    mdiProjectStopWordList.append('URL')
    
    #Reading stopword file
    mdiProjectStopWordFile = open(mdiProjectStopWordListFileName, 'r')
    
    #Reading each line of the stopword file and put into a list
    mdiProjectStopWordLine = mdiProjectStopWordFile.readline()
    
    #Readin each line and populating a list with stop words
    while mdiProjectStopWordLine:
        
        midProjectWord = mdiProjectStopWordLine.strip()
        mdiProjectStopWordList.append(midProjectWord)
        
        mdiProjectStopWordLine = mdiProjectStopWordFile.readline()
    
    #Close stop word file
    mdiProjectStopWordFile.close()
    
    return mdiProjectStopWordList

#Create feature vector based on a single tweet.
def mdiProjectGetFeatureVectorForSingleTweet(mdiProjectTweet, mdiProjectStopWordsList):
    
    mdiProjectFeatureVector = []
    
    #Split the tweet into words
    mdiProjectTweetWordList = mdiProjectTweet.split()
    
    for mdiProjectWord in mdiProjectTweetWordList:
        
        #Replace two or more with two occurrences
        mdiProjectWord = mdiProjectReplaceTwoOrMore(mdiProjectWord)
        
        #Strips the punctuations
        mdiProjectWord = mdiProjectWord.strip('\'"?,.')
        
        #Check if the word start with an alphabet
        mdiProjectVal = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", mdiProjectWord)
        
        #Ignore if it is a stop words
        if(mdiProjectWord in mdiProjectTweetWordList or mdiProjectVal is None ):
            
            continue
        else:
            mdiProjectFeatureVector.append(mdiProjectWord)
    
    return mdiProjectFeatureVector


def mdiProjectExtractFeatures(mdiProjectTweetList): 
    
    mdiProjectSampleTweetList = mdiProjectReadSampleFile(mdiProjectReadFiles("mdiProjectFiles","SampleTrainingData.csv"))
    mdiProjectTrainingSet, mdiProjectTestingSet = mdiProjectSplitTrainTest(mdiProjectSampleTweetList)
    mdiProjectStopWordsList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles","StopWords.txt"))
    
    #Getting features from the tweets
    mdiProjectTweetsList, mdiProjectFeatureList = mdiProjectGetFeatureListAndTweetListForTrainingSet(mdiProjectTrainingSet, mdiProjectStopWordsList)
    
    #Getting features from the tweets
    #mdiProjectTweetsList, mdiProjectFeatureList = mdiProjectGetFeatureListAndTweetListForTrainingSet(mdiProjectReadFiles("mdiProjectFiles","SampleTrainingData.csv"), mdiProjectReadFiles("mdiProjectFiles","StopWords.txt")) 

    mdiProjectTweetWordsSet = set(mdiProjectTweetList)
    
    mdiProjectFeatures = {}
    
    for mdiProjectWord in mdiProjectFeatureList:
        
        #Search feature words in mdiProjectTweetWordsSet and populates features vectors
        mdiProjectFeatures['contains(%s)' % mdiProjectWord] = (mdiProjectWord in mdiProjectTweetWordsSet)
    
    return mdiProjectFeatures


def mdiProjectReadFiles(mdiProjectFolderName, mdiProjectFileName):
    d = os.getcwd()
    d1 = os.path.join(d, mdiProjectFolderName)
    fname = os.path.join(d1, mdiProjectFileName)
    return fname
    
def mdiProjectGetFeatureListAndTweetListForTrainingSet(mdiProjectTweets, mdiProjectStopWordsList):
    
    #Read tweet from the training set tweet file
    #mdiProjectTweets = csv.reader(open('files/SampleTrainingData.csv', 'r'), delimeter = ',', quotechar='|')
    #mdiProjectTweets = csv.reader(open(mdiProjectTrainingSetCSV, 'r', encoding='ISO-8859-1'))
    #mdiProjectTweets = mdiProjectReadSampleFile(mdiProjectTrainingSetCSV)
    
    #Read stopword file and populate them in a list
    #mdiProjectStopWordsList = mdiProjectStopWordList(mdiProjectStopWordsFile)
    
    mdiProjectFeatureList = []
    
    #Get the tweet words
    mdiProjectTweetsList = []
    
    for mdiProjectTweetRow in mdiProjectTweets:
        
        mdiProjectSentiment = mdiProjectTweetRow[0]
        mdiProjectTweet = mdiProjectTweetRow[1]
        
        mdiProjectProcessedTweet = mdiProjectProcessTweet(mdiProjectTweet)
        mdiProjectFeatureVector = mdiProjectGetFeatureVectorForSingleTweet(mdiProjectProcessedTweet, mdiProjectStopWordsList)
        
        mdiProjectFeatureList.extend(mdiProjectFeatureVector)
        
        mdiProjectTweetsList.append((mdiProjectFeatureVector, mdiProjectSentiment))
    
    #Remove feature duplicates from feature list
    mdiProjectFeatureList = list(set(mdiProjectFeatureList))

    return mdiProjectTweetsList, mdiProjectFeatureList

def mdiProjectReadSampleFile(mdiProjectTrainingSetCSV):
    
    mdiProjectTweets = csv.reader(open(mdiProjectTrainingSetCSV, 'r', encoding='ISO-8859-1'))
    
    return mdiProjectTweets

def mdiProjectSplitTrainTest(mdiProjectTweets):
    
    mdiProjectTweets = list(mdiProjectTweets)
    mdiProjectTrainingSet, mdiProjectTestingSet = train_test_split(mdiProjectTweets, test_size=0.3, random_state=200)
    
    return mdiProjectTrainingSet, mdiProjectTestingSet
    
if __name__ == "__main__":
    mdiProjectSampleTweetList = mdiProjectReadSampleFile(mdiProjectReadFiles("mdiProjectFiles","SampleTrainingData.csv"))
    mdiProjectTrainingSet, mdiProjectTestingSet = mdiProjectSplitTrainTest(mdiProjectSampleTweetList)
    mdiProjectStopWordsList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles","StopWords.txt"))
    mdiProjectGetFeatureListAndTweetListForTrainingSet(mdiProjectTrainingSet, mdiProjectStopWordsList)
    



    
    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    