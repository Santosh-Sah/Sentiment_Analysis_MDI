# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:27:19 2019

@author: Santosh Sah
"""
import time
import zipfile
import pandas as pd
from io import BytesIO
import pickle
from flask import Flask, request, make_response, send_file
from MDIProjectUtils import mdiProjectCleanTweet
from flasgger import Swagger
from MDIProjectSentimentAnalysis import (mdiProjectExtractFeatures, mdiProjectGetFeatureVectorForSingleTweet, 
                                         mdiProjectProcessTweet, mdiProjectStopWordList, mdiProjectReadFiles)
from MDIProjectTweeterUtils import (mdiProjectSearchTweetBasedOnSearchTerm,mdiProjectTweeterAuthHandler, mdiProjectGetNormalizedTweeterConfig,
                                    mdiProjectProcessTweetJsonFile)

app = Flask(__name__)
swagger = Swagger(app)
    
@app.route("/predict_sentiment_single_tweet_using_Naive_Bayes", methods=['POST'])
def sentimentsAnalysisSingleTweetUsingNaiveBayes():
    
    """ Predict sentiments of a tweet
    ---
    produces:
        - "application/xml"
        - "application/json"
    parameters:
        - name: tweet
          in: query
          type: string
          required: true
    responses:
        content:
            application/json:
                schema:
                    type: object
    """
    
    #Open NaiveBayesClassifier picke file
    with open("NaiveBayesClassifierModel.pkl","rb") as NaiveBayesClassifier:
        NaiveBayesClassifierModel = pickle.load(NaiveBayesClassifier)
        
    #Get the tweet from UI
    mdiProjectTweet = request.args.get("tweet")
    
    #Process the tweet
    porcessedTweet = mdiProjectProcessTweet(mdiProjectTweet)
    
    #Get all the stop words
    stopWordList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles", "StopWords.txt"))
    
    #Get the features vector for a single tweet
    featureVector = mdiProjectGetFeatureVectorForSingleTweet(porcessedTweet, stopWordList)
    
    #Get the feacture words
    featureWords = mdiProjectExtractFeatures(featureVector)
    
    #Get sentiments based on the feature words
    tweetSentiment = NaiveBayesClassifierModel.classify(featureWords)
    
    return tweetSentiment

@app.route("/predict_sentiment_tweet_topic_using_naive_bayes", methods=['POST'])
def sentimentsAnalysisBasedOnTweetTopicUsingNaiveBayes():
    
    """ Predict sentiments of a tweet
    ---
    parameters:
        - name: tweetTopic
          in: query
          type: string
          required: true
    responses:
        content:
            application/json:
                schema:
                    type: object
    """
    #Open NaiveBayesClassifier picke file
    with open("NaiveBayesClassifierModel.pkl","rb") as NaiveBayesClassifier:
        NaiveBayesClassifierModel = pickle.load(NaiveBayesClassifier)
    
    #Get the tweet topic
    mdiProjectTweetTopic = request.args.get("tweetTopic")
    
    #Get the tweets based on the tweet topic and save in file
    mdiProjectSearchTweetBasedOnSearchTerm(mdiProjectTweeterAuthHandler(mdiProjectGetNormalizedTweeterConfig("mdiProjectFiles/tweeterConfig.json")), mdiProjectTweetTopic)
    
    #Process saves tweets json file
    mdiProjectTweetTextList, mdiProjectTweetText = mdiProjectProcessTweetJsonFile("mdiProjectTweets.json")  
    
    #Get all the stop words
    stopWordList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles", "StopWords.txt"))
    
    sentiments = []
    
    for mdiProjectTweet in mdiProjectTweetTextList:
        
        #Process the tweet
        porcessedTweet = mdiProjectProcessTweet(mdiProjectTweet)
        
        #Get the features vector for a single tweet
        featureVector = mdiProjectGetFeatureVectorForSingleTweet(porcessedTweet, stopWordList)
        
        #Get the feacture words
        featureWords = mdiProjectExtractFeatures(featureVector)
        
        #Get sentiments based on the feature words
        tweetSentiment = NaiveBayesClassifierModel.classify(featureWords)
        tweetSentiment = tweetSentiment.replace("\"","")
        
        sentiments.append(tweetSentiment)
    
    mdiProjectTweetText["sentiments"] = sentiments
    
    #Makes an excel file with sentimets and make it downloadable
    mdiProjectOutput = BytesIO()
    mdiProjectOutputWriter = pd.ExcelWriter(mdiProjectOutput, engine="xlsxwriter")
    mdiProjectTweetText.to_excel(mdiProjectOutputWriter, sheet_name="twitter_sentiments", encoding="utf-8", index=False)
    mdiProjectOutputWriter.save()
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        names = ['twitter_sentiments.xlsx']
        files = [mdiProjectOutput]
        for i in range(len(files)):
            data = zipfile.ZipInfo(names[i])
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, files[i].getvalue())
    memory_file.seek(0)
    response = make_response(send_file(memory_file, attachment_filename='twitter_sentiments.zip',
                                       as_attachment=True))
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    return response

@app.route("/predict_sentiment_single_tweet_using_Logistics", methods=['POST'])
def sentimentsAnalysisSingleTweetUsingLogisticRegression():
    
    """ Predict sentiments of a tweet
    ---
    produces:
        - "application/xml"
        - "application/json"
    parameters:
        - name: tweet
          in: query
          type: string
          required: true
    responses:
        content:
            application/json:
                schema:
                    type: object
    """
    
    with open("mdiProjectLogisticRegressionclassifier.pkl","rb") as mdiProjectLogisticRegressionclassifier:
        mdiProjectLogisticRegressionclassifierModel = pickle.load(mdiProjectLogisticRegressionclassifier)
    
    with open("mdiProjectTfidfvectorizer.pkl","rb") as mdiProjectTfidfvectorizer_Pickle:
        mdiProjectTfidfvectorizer = pickle.load(mdiProjectTfidfvectorizer_Pickle)
    
    #Get the tweet from UI
    mdiProjectTweet = request.args.get("tweet")
        
    #mdiProjectTweetList = ["This product is bad"]
    mdiProjectTweetList = [mdiProjectTweet]
    mdiProjectTweetList = mdiProjectCleanTweet(mdiProjectTweetList)
    
    sample = mdiProjectTfidfvectorizer.transform(mdiProjectTweetList).toarray()
    
    sentiments = mdiProjectLogisticRegressionclassifierModel.predict(sample)
    
    return sentiments

@app.route("/predict_sentiment_tweet_topic_using_Logistics", methods=['POST'])
def sentimentsAnalysisBasedOnTweetTopicUsingLogisticRegression():
    
    """ Predict sentiments of a tweet
    ---
    parameters:
        - name: tweetTopic
          in: query
          type: string
          required: true
    responses:
        content:
            application/json:
                schema:
                    type: object
    """
    
    with open("mdiProjectLogisticRegressionclassifier.pkl","rb") as mdiProjectLogisticRegressionclassifier:
        mdiProjectLogisticRegressionclassifierModel = pickle.load(mdiProjectLogisticRegressionclassifier)
    
    with open("mdiProjectTfidfvectorizer.pkl","rb") as mdiProjectTfidfvectorizer_Pickle:
        mdiProjectTfidfvectorizer = pickle.load(mdiProjectTfidfvectorizer_Pickle)
    
    #Get the tweet topic
    mdiProjectTweetTopic = request.args.get("tweetTopic")
    
    #Get the tweets based on the tweet topic and save in file
    mdiProjectSearchTweetBasedOnSearchTerm(mdiProjectTweeterAuthHandler(mdiProjectGetNormalizedTweeterConfig("mdiProjectFiles/tweeterConfig.json")), mdiProjectTweetTopic)
    
    #Process saves tweets json file
    mdiProjectTweetTextList, mdiProjectTweetText = mdiProjectProcessTweetJsonFile("mdiProjectTweets.json") 
    
    mdiProjectTweetList = mdiProjectCleanTweet(mdiProjectTweetTextList)
    
    sample = mdiProjectTfidfvectorizer.transform(mdiProjectTweetList).toarray()
    
    sentiments = mdiProjectLogisticRegressionclassifierModel.predict(sample)
    
    mdiProjectTweetText["sentiments"] = sentiments
    
    #Makes an excel file with sentimets and make it downloadable
    mdiProjectOutput = BytesIO()
    mdiProjectOutputWriter = pd.ExcelWriter(mdiProjectOutput, engine="xlsxwriter")
    mdiProjectTweetText.to_excel(mdiProjectOutputWriter, sheet_name="twitter_sentiments", encoding="utf-8", index=False)
    mdiProjectOutputWriter.save()
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        names = ['twitter_sentiments.xlsx']
        files = [mdiProjectOutput]
        for i in range(len(files)):
            data = zipfile.ZipInfo(names[i])
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, files[i].getvalue())
    memory_file.seek(0)
    response = make_response(send_file(memory_file, attachment_filename='twitter_sentiments.zip',
                                       as_attachment=True))
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    return response
        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
    #sentimentsAnalysisSingleTweetUsingNaiveBayes()