# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:52:08 2019

@author: Santosh Sah
"""
import time
import pandas as pd
import json
import os
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

#Normalize Tweeter configuration file
def mdiProjectGetNormalizedTweeterConfig(mdiProjectTweetConfigJson):
    
    with open(mdiProjectTweetConfigJson) as configFile:
        
        configFileData = json.load(configFile)
    
    #Normalize configuration file
    mdiProjectTweeterConfig = pd.io.json.json_normalize(configFileData)
    
    return mdiProjectTweeterConfig

def mdiProjectTweeterAuthHandler(mdiProjectTweeterConfig):
    
    #Getting tweeter configuration
    mdiProjectTweeterConsumerKey = mdiProjectTweeterConfig.consumer_key[0].encode("ascii")
    mdiProjectTweeterConsumerSecret = mdiProjectTweeterConfig.consumer_secret[0].encode("ascii")
    mdiProjectTweeterAccessToken = mdiProjectTweeterConfig.access_token[0].encode("ascii")
    mdiProjectTweeterAccessTokenSecret = mdiProjectTweeterConfig.access_token_secret[0].encode("ascii")
    
    #Create OAuthHandler with the help of consumer key, consumer secret, access token and access token secret
    mdiProjectoAuthHandler = OAuthHandler(mdiProjectTweeterConsumerKey, mdiProjectTweeterConsumerSecret)
    mdiProjectoAuthHandler.set_access_token(mdiProjectTweeterAccessToken, mdiProjectTweeterAccessTokenSecret)
    
    return mdiProjectoAuthHandler
    
#Create Tweeter Listener and save the tweets in a file based on the topic
class MdiProjectTweeterListner(StreamListener):
    
    #Time limit to get the tweets is for 10 seconds
    def __init__(self, time_limit = 10):
        
        self.start_time = time.time()
        self.limit = time_limit
        
        try:
            #Remove mdiProjectTweets.json if already exit
            os.remove("mdiProjectTweets.json")
            self.saveFile = open("mdiProjectTweets.json","a")
        except BaseException as exception:
            print("Error on_data: %s" % str(exception)) 
            
        self.saveFile = open("mdiProjectTweets.json","a")
        super(MdiProjectTweeterListner, self).__init__()
        
    def on_data(self, data):
        
        if(time.time() - self.start_time) < self.limit:
            
            self.saveFile.write(data)
            self.saveFile.write("\n")
            return True
        else:
            self.saveFile.close()
            return False
    
    def on_error(self, status):
        print(status)
        return True

def mdiProjectSearchTweetBasedOnSearchTerm(mdiProjectOAuthHandler, tweetTopic):
    
    try:
        #Create stream object to communicate with tweeter
        mdiProjectTweeterStream = Stream(mdiProjectOAuthHandler, MdiProjectTweeterListner())
        
        #Filter tweets based on the topic
        mdiProjectTweeterStream.filter(track=[tweetTopic])
    except BaseException as exception:
        print(exception)

def mdiProjectProcessTweetJsonFile(mdiProjectTweetJsonFile):
    
    with open(mdiProjectTweetJsonFile, encoding="utf-8") as tweetJsonFile:
        data = tweetJsonFile.readlines()
    
    data = list(map(lambda x: x.rstrip(), data))
    
    #Remove empty tweets from list
    data = [i for i in data if i]
    
    dataJosnStr = "[" + ','.join(data) + "]"
    
    dataJsonStrDF = pd.read_json(dataJosnStr)
    
    mdiProjectTweetText= dataJsonStrDF[["text"]]
    mdiProjectTweetTextList = mdiProjectTweetText["text"]
    
     
    return mdiProjectTweetTextList, mdiProjectTweetText
        
if __name__ == "__main__":
    
    #mdiProjectGetTweetsAsPandaDataFrame(mdiProjectTWPython("mdiProjectFiles/tweeterConfig.json"), mdiProjectTWPythonQuery)
    #mdiProjectSearchTweetBasedOnSearchTerm(mdiProjectTweeterAuthHandler(mdiProjectGetNormalizedTweeterConfig("mdiProjectFiles/tweeterConfig.json")))
    mdiProjectProcessTweetJsonFile("mdiProjectTweets.json")
   
  
    