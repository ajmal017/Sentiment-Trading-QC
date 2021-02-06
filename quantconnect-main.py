from QuantConnect.Data.Custom.Tiingo import *
from datetime import datetime, timedelta
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import unicodedata

class TiingoNewsSentimentAlgorithm(QCAlgorithm):
    
    filteredByPrice = None

    def Initialize(self):
        #sets start/end date
        #sets starting cash
        #adds the universe selection (described below)
        
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2020, 5, 30)
        #self.SetStartDate(2017, 1, 1)
        #self.SetEndDate(2017, 6, 30)
        self.SetCash(100000)  
        self.AddUniverse(self.CoarseSelectionFilter)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetAlpha(NewsSentimentAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel()) 
        #trade portfolio value divided by number of insights
        #Immediate Execution Model uses market orders to immediately fill algorithm portfolio targets
        self.SetExecution(ImmediateExecutionModel()) 
        self.SetRiskManagement(NullRiskManagementModel())
        
    def CoarseSelectionFilter(self, coarse):
        #takes stocks that are valued at higher than $10
        #orders based on price, chooses top 10 most expensive
        sortedByDollarVolume = sorted(coarse, key=lambda c: c.DollarVolume, reverse=True)
        filteredByPrice = [c.Symbol for c in sortedByDollarVolume if c.Price > 10]
        return filteredByPrice[:10] 

class NewsData():
    def __init__(self, symbol):
        self.Symbol = symbol
        # Configue the window to accept 100 data points and save to Window
        self.Window = RollingWindow[float](100)  
        
class NewsSentimentAlphaModel(AlphaModel):
    
    def __init__(self): 
        # Storage for our data class
        self.newsData = {}
        self.wordScores = {
        "bad":-0.5, "good":0.25, "negat":-0.125, "fail":-0.25,
        "posit":0.625,"great":0.01,
        "success":0.125,"profit":0.75}
        
        #bad .125
        #negat .032
        #good .066
        #fail .066
        #posit .015
        #great effectively 0
        #success .034
        #profit .171
        #words chosen based on tiingo news and other sources
        #stemmed in different python file
        #weights are all the same, chosen based on the bootcamp Tiingo file
        #weights presented in class were all the same, weights in this file are based on regressions in notebook
        #some words had r squared of 0, so those were deleted
                
    def Update(self, algorithm, data):
        stemmer = SnowballStemmer("english")
        insights = []
        news = data.Get(TiingoNews) 
    
        for article in news.Values:
            #ran into issues on 6/4/2020 with article.Description...
            #used try/except to set score = 0 on that day
            #turns the article into a lowercase list of words, stems that list, then sums the score (number of occurances) of the words in the article
            #that also match the words in self.wordScores 
            #ie: score + 0.5 if posit appears in the article...
            try:
                words = article.Description.lower().split(" ")
                words = [stemmer.stem(word) for word in words]
                score = sum([self.wordScores[word] for word in words
                    if word in self.wordScores])
            except:
                score = 0
                
            #get the underlying symbol and save to the variable symbol
            symbol = article.Symbol.Underlying
            
            #add scores to the rolling window associated with its newsData symbol
            self.newsData[symbol].Window.Add(score)
            
            #sum the rolling window scores, save to sentiment
            #if sentiment aggregate score for the time period is greater than 2, emit an up insight
            #if below 2, emit down insight
            #used 2 in our presentation, changed to a different amount due to reducing a lot of values in the self.wordscores
            sentiment = sum(self.newsData[symbol].Window)
            if sentiment >= 2.5:
                insights.append(Insight.Price(symbol, timedelta(1), InsightDirection.Up))
            elif sentiment <= -1.5:
                insights.append(Insight.Price(symbol, timedelta(1), InsightDirection.Down))
        return insights
    
    def OnSecuritiesChanged(self, algorithm, changes):
    #adds/remove data requested when a security leaves the universe
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            newsAsset = algorithm.AddData(TiingoNews, symbol)
            #create a new instance of the NewsData() and store in self.newsData[symbol]
            self.newsData[symbol] = NewsData(newsAsset.Symbol)
            
        #remove news data once assets are removed from our universe
        for security in changes.RemovedSecurities:
            newsData = self.newsData.pop(security.Symbol, None)
            if newsData is not None:
                algorithm.RemoveSecurity(newsData.Symbol)