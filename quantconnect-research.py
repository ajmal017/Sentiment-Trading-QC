#!/usr/bin/env python
# coding: utf-8

# ![QuantConnect Logo](https://cdn.quantconnect.com/web/i/icon.png)
# <hr>



# In[15]:


from QuantConnect.Data.Custom.Tiingo import *
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt

# QuantBook Analysis Tool 
# For more information see [https://www.quantconnect.com/docs/research/overview]
qb = QuantBook()
qb.start_time = datetime(2017, 1, 1)
qb.end_time = datetime(2017, 6, 1)

sp500 = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 
         'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 
         'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 
         'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 
         'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AIV', 'AAPL', 
         'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 
         'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 
         'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BF.B', 
         'CHRW', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 
         'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CERN', 'CF', 'SCHW', 'CHTR', 'CVX', 
         'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 
         'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 
         'CPRT', 'GLW', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 
         'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 
         'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 
         'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 
         'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 
         'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FLIR', 'FLS', 
         'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 
         'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 
         'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX', 'HD', 'HON', 
         'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 
         'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 
         'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 
         'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LB', 'LHX', 'LH', 'LRCX', 
         'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 
         'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 
         'MKC', 'MXIM', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 
         'MAA', 'MHK', 'TAP', 'MDLZ', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NOV', 
         'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NSC', 'NTRS', 
         'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC', 
         'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PBCT', 
         'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 
         'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 
         'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 
         'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 
         'NOW', 'SHW', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 
         'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 
         'FTI', 'TDY', 'TFX', 'TER', 'TXN', 'TXT', 'TMO', 'TIF', 'TJX', 'TSCO', 'TT', 'TDG', 
         'TRV', 'TFC', 'TWTR', 'TYL', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 
         'UNH', 'UPS', 'URI', 'UHS', 'UNM', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 
         'VFC', 'VIAC', 'VTRS', 'V', 'VNT', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS', 
         'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 
         'WLTW', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

qb.wordScores = {
            "bad": -0.5, "good": 0.5, "negative": -0.5, 
            "great": 0.5, "growth": 0.5, "fail": -0.5, 
            "failed": -0.5, "success": 0.5, "nailed": 0.5,
            "beat": 0.5, "missed": -0.5, "profitable": 0.5,
            "beneficial": 0.5, "right": 0.5, "positive": 0.5, 
            "large":0.5, "attractive": 0.5, "sound": 0.5, 
            "excellent": 0.5, "wrong": -0.5, "unproductive": -0.5, 
            "lose": -0.5, "missing": -0.5, "mishandled": -0.5, 
            "un_lucrative": -0.5, "up": 0.5, "down": -0.5,
            "unproductive": -0.5, "poor": -0.5, "wrong": -0.5,
            "worthwhile": 0.5, "lucrative": 0.5, "solid": 0.5
        }

qb.nonStemScores = {
                "bad": -0.5, "good": 0.5, "negative": -0.5, "fail": -0.5,
                "terrible": -0.5,"unprofitable": -0.5,"positive": 0.5,"great": 0.5,
                "success": 0.5,"profitable": 0.5
                }

qb.stemScores = {
                "bad": -0.5, "good": 0.5, "negat": -0.5, "fail": -0.5,
                "terribl": -0.5,"unprofit": -0.5,"posit": 0.5,"great": 0.5,
                "success": 0.5,"profit": 0.5
                }

qb.posStemScores = {
                "good": 0.5,"posit": 0.5,"great": 0.5,
                "success": 0.5,"profit": 0.5
                }

qb.negStemScores = {
                "bad": -0.5, "negat": -0.5, "fail": -0.5,
                "terribl": -0.5,"unprofit": -0.5
                }
qb.stembad = {"bad": 2}
qb.stemgood = {"good": 2}
qb.stemnegat = {"negat": 2}
qb.stemfail = {"fail": 2}
qb.stemterribl = {"terribl": 2}
qb.stemunprofit = {"unprofit": 2}
qb.stemposit = {"posit": 2}
qb.stemgreat = {"great": 2}
qb.stemsuccess = {"success": 2}
qb.stemprofit = {"profit": 2}

def getScores(ticker):
    #adds equity based on ticker to QuantBook, defines symbol
    symbol = qb.AddEquity(ticker).Symbol
    #converts to ticker.TiingoNews
    news = qb.AddData(TiingoNews, symbol).Symbol
    #loads history of various articles between start and end time to QuantBook, also sets resolution
    historyNews = qb.History(TiingoNews, news, qb.start_time, qb.end_time, Resolution.Daily)
    
    print('symbol ', symbol)
    print('news ', news)


    scores = []
    dates = []
    negScores = []
    posScores = []
    totalWords = []
    nonStemScores = []
    stembad = []
    stemgood = []
    stemnegat = []
    stemfail = []
    stemterribl = []
    stemunprofit = []
    stemposit = []
    stemgreat = []
    stemsuccess = []
    stemprofit = []
    print('historyNews shape ', historyNews.shape)
    #goes through articles in history
    for article in historyNews:
        #separates description from other article details
        #print('history is ', history)
        description = historyNews.reset_index(level=0)['description']
        #makes all descriptions lower case, splits based on spaces
        words = description.str.lower()
        words = words.str.split(" ")
        #converts from series to dataframe
        wordDF = words.to_frame()
        qb.score = []
        #goes through wordList for each article - list of words in article, lower case
        for wordList in wordDF['description']:
            #SnowBallStemmer from NLTK package stems words in wordList from each article
            snowBallStemmer = SnowballStemmer("english")
            stemmedList = [snowBallStemmer.stem(word) for word in wordList]
            #print('stemmed list ', stemmedList)
            
            #sums scores for each article description list of words
            #scoreAmt = sum([qb.wordScores[word] for word in wordList
                #if word in qb.wordScores])
            
            #score of stemmed words and stemmed article - both pos and neg words
            scoreAmt = sum([qb.stemScores[word] for word in stemmedList
                if word in qb.stemScores])
            
            #scores only on positive stemmed words
            posScoreAmt = sum([qb.posStemScores[word] for word in stemmedList
                if word in qb.posStemScores])
            
            #scores only on negative stemmed words
            negScoreAmt = sum([qb.negStemScores[word] for word in stemmedList
                if word in qb.negStemScores])
            
            #counts stemmed word appearances
            totalNumWords = sum([1 for word in stemmedList
                if word in qb.stemScores])
            
            #scores article based on non stemmed words - both pos and neg words
            nonStemScoreAmt = sum([qb.nonStemScores[word] for word in wordList
                if word in qb.nonStemScores])
            
            stembadscore = sum([qb.stembad[word] for word in stemmedList if word in qb.stembad])
            
            stemnegatscore = sum([qb.stemnegat[word] for word in stemmedList if word in qb.stemnegat])
            
            stemgoodscore = sum([qb.stemgood[word] for word in stemmedList if word in qb.stemgood])
            
            stemfailscore = sum([qb.stemfail[word] for word in stemmedList if word in qb.stemfail])
            
            stemterriblscore = sum([qb.stemterribl[word] for word in stemmedList 
                                    if word in qb.stemterribl])
            
            stemunprofitscore = sum([qb.stemunprofit[word] for word in stemmedList 
                                     if word in qb.stemunprofit])
            
            stempositscore = sum([qb.stemposit[word] for word in stemmedList if word in qb.stemposit])
            
            stemgreatscore = sum([qb.stemgreat[word] for word in stemmedList if word in qb.stemgreat])
            
            stemsuccessscore = sum([qb.stemsuccess[word] for word 
                                    in stemmedList if word in qb.stemsuccess])
            
            stemprofitscore = sum([qb.stemprofit[word] for word in stemmedList if word in qb.stemprofit])
            
            #appends score into qb.score
            qb.score.append(scoreAmt)
            #gets dates used to index the dataframe with the wordLists of each article
            indx = wordDF.index
            #converts to list of dates
            listNames = list(indx)
            #current row index number can be recorded from the length of qb.score, as all scores are appended
            #to qb.score regardless of magnitude
            index = len(qb.score)
            
            #checks to make sure article score is significantly positive or negative
            if scoreAmt >= 2 or scoreAmt <= -2:
                #scores list only holds scores bigger than 2 or smaller than -2
                scores.append(scoreAmt)
                #print('col 0 is ', wordDF.info)
                
                #uses row index number to get the date of the article, appends to list
                date = listNames[index]
                dates.append(date)
                
                negScores.append(negScoreAmt)
                posScores.append(posScoreAmt)
                totalWords.append(totalNumWords)
                nonStemScores.append(nonStemScoreAmt)
                stembad.append(stembadscore)
                stemnegat.append(stemnegatscore)
                stemgood.append(stemgoodscore)
                stemfail.append(stemfailscore)
                stemterribl.append(stemterriblscore)
                stemunprofit.append(stemunprofitscore)
                stemposit.append(stempositscore)
                stemgreat.append(stemgreatscore)
                stemsuccess.append(stemsuccessscore)
                stemprofit.append(stemprofitscore)
                #print('date is ', listNames[index].day, ' month ', listNames[index].month, ' year ', listNames[index].year)
                #print('date is ', date)
        #adds extra column to dataframe for score of the article description      
        #print('scores ', scores)
        #print('dates ', dates)
        
        #structures data to go into dataframe, gives column names and respective list
        data = {'scores':scores, 'posScores':posScores, 'negScores':negScores, 'totalWords':totalWords, 
                'nonStemScores':nonStemScores,"stembad":stembad, "stemnegat":stemnegat, "stemgood": stemgood,
                "stemfail":stemfail, "stemterribl":stemterribl, "stemunprofit":stemunprofit, 
                "stemposit":stemposit, "stemgreat":stemgreat, "stemsuccess":stemsuccess,
                "stemprofit":stemprofit,
        'dates':dates} 
        #bigScores becomes dataframe with data for each article that scored significantly
        bigScores = pd.DataFrame(data)
        print('bigScores ', bigScores)
        print('bigScores shape ', bigScores.shape)
        print('bigScores info ', bigScores.info)
        wordDF['score'] = qb.score
        print('words ', wordDF)
        #print('cols ')
        #for col in wordDF.columns:
           # print(col)
        print(wordDF.shape)
        #returns bigScores dataframe which has a variety of different scoring techniques for each article date
        return bigScores



#function gets historical price data for a ticker and computes regressions based on data collected
#takes parameters of a dataframe returned from getScores and a ticker string
#uses dates of articles with very high/low scores to match to close price data

def regress_word(df, ticker):
    
    #get history of prices 
    symbol = qb.AddEquity(ticker).Symbol
    historyPrice = qb.History(symbol, qb.start_time, qb.end_time, Resolution.Daily)
    
    #print('cols ')
    #for col in historyPrice.columns:
          # print(col)
        
    #gets date indexes in historyPrice dataframe
    indx = historyPrice.index
    #converts dates into a list
    listNames = list(indx)
    timestamps = []
    #loops through list of dates to remove unnecessary data, only keeps timestamp - includes day and time
    for a in range(len(listNames)):
        timestamps.append(listNames[a][1])
    
    #print('timestamps ', timestamps)
    #print('timestamp 0 ', timestamps[0].day, ' ' , timestamps[0].month, ' ', timestamps[0].year)
    
    priceChange = []
    newScores = []
    newPosScores = []
    newNegScores = []
    newTotalWords = []
    newNonStemScores = []
    newstembad = []
    newstemnegat = []
    newstemgood = []
    newstemfail = []
    newstemterribl = []
    newstemunprofit = []
    newstemposit = []
    newstemgreat = []
    newstemsuccess = []
    newstemprofit = []
    
    #cycles through rows in dataframe passed as a parameter to the function
    for date in df['dates']:
        #print('original date ', date)
        
        #calculates the day after the date, to get day after article is published
        afterDate = date + timedelta(days=1)
        #print('after date ', afterDate)
        
        #converts from date object to timestamp object to match the times provided in the historyPrice dataframe
        afterDateTS =  pd.Timestamp(datetime(afterDate.year, afterDate.month, afterDate.day))
        
        #print('timestamp convert ', afterDateTS)
        #checks to see if the day after the article is published exists in the historyPrice dataframe
        if afterDateTS in timestamps:
            #gets the row index where the original date appears
            rowIndexDF = df[df['dates']==date].index[0]
            #print('index is ' , rowIndexDF)
            #print('new scores ', newScores)
            
            #gets index of where the day+1 appears in the timestamps from historyPrice dataframe
            indexDate = timestamps.index(afterDateTS)
            #print('close price ', historyPrice.loc[ticker]["close"])
            
            #creates a list from the close prices for the stock from historyPrice dataframe
            closeList = (historyPrice.loc[ticker]["close"]).tolist()
            
            #print('close list ', closeList)
            #print('close price is ', closeList[indexDate])
            
            #makes sure index of the date does not equal zero, so it can grab the day before day+1 data
            if indexDate != 0:
                #rowIndexDF is index of day article is published
                #appends data from the rowIndexDF and the respective column that data is stored in
                newScores.append(df.iloc[rowIndexDF, 0])
                newPosScores.append(df.iloc[rowIndexDF, 1])
                newNegScores.append(df.iloc[rowIndexDF, 2])
                newTotalWords.append(df.iloc[rowIndexDF, 3])
                newNonStemScores.append(df.iloc[rowIndexDF, 4])
                newstembad.append(df.iloc[rowIndexDF,5])
                newstemnegat.append(df.iloc[rowIndexDF,6])
                newstemgood.append(df.iloc[rowIndexDF,7])
                newstemfail.append(df.iloc[rowIndexDF,8])
                newstemterribl.append(df.iloc[rowIndexDF,9])
                newstemunprofit.append(df.iloc[rowIndexDF,10])
                newstemposit.append(df.iloc[rowIndexDF,11])
                newstemgreat.append(df.iloc[rowIndexDF,12])
                newstemsuccess.append(df.iloc[rowIndexDF,13])
                newstemprofit.append(df.iloc[rowIndexDF,14])
                #calculates the price change to the following day
                #day after article price - day of article price = change in price
                priceChange.append(closeList[indexDate]-closeList[indexDate-1])
        else:
            #if day+1 does not appear in the priceHistory
            print('not in history')
    #print('all price changes ', priceChange)
    print('length price changes ', len(priceChange))
    print('length of scores ', len(newScores))
    print('length of pos scores ', len(newPosScores))
    print('length of neg scores ', len(newNegScores))
    print('length of total words ', len(newTotalWords))
    print('length of non stems scores ', len(newNonStemScores))
    print('newnonstemscores ', newNonStemScores)
    
    #just reshapes the list of data to be used in the LinearRegression function, makes it a 2D array
    newScoresX = np.reshape(newScores, (-1,1))
    newPosScoresX = np.reshape(newPosScores, (-1,1))
    newNegScoresX = np.reshape(newNegScores, (-1,1))
    newTotalWordsX = np.reshape(newTotalWords, (-1,1))
    newNonStemScoresX = np.reshape(newNonStemScores, (-1,1))
    priceChangeY = np.reshape(priceChange, (-1,1))
    newstembadx = np.reshape(newstembad, (-1,1))
    newstemnegatx = np.reshape(newstemnegat, (-1,1))
    newstemgoodx = np.reshape(newstemgood, (-1,1))
    newstemfailx = np.reshape(newstemfail, (-1,1))
    newstemterriblx = np.reshape(newstemterribl, (-1,1))
    newstemunprofitx = np.reshape(newstemunprofit, (-1,1))
    newstempositx = np.reshape(newstemposit, (-1,1))
    newstemgreatx = np.reshape(newstemgreat, (-1,1))
    newstemsuccessx = np.reshape(newstemsuccess, (-1,1))
    newstemprofitx = np.reshape(newstemprofit, (-1,1))
    
    #combines lists in Cartesian product using zip function
    twoFactors = list(zip(newScores, newNegScores))
    print('zip two factors ', (twoFactors))
    threeFactors = list(zip(newScores, newNegScores, newTotalWords))
    print('zip three factors ', (threeFactors))
    
    threeFactors3 = list(zip(newScores, newNegScores, newNonStemScores))
    #print('new scores x ', newScoresX)
    #print('price change y ', priceChangeY)
    #regress word score as x on price change as y
    
    #prints out regression data for following regressions
    print('ticker ', ticker)
    reg1 = LinearRegression().fit(newScoresX, priceChangeY)
    print('reg1 - wordScores')
    print('r squared ', reg1.score(newScoresX, priceChangeY))
    print('coeff ', reg1.coef_)
    print('intercept ', reg1.intercept_)
    
    
    reg2 = LinearRegression().fit(newPosScoresX, priceChangeY)
    print('reg2 - posWordScores')
    print('r squared ', reg2.score(newPosScoresX, priceChangeY))
    print('coeff ', reg2.coef_)
    print('intercept ', reg2.intercept_)
    
    reg3 = LinearRegression().fit(newNegScoresX, priceChangeY)
    print('reg3 - negWordScores')
    print('r squared ', reg3.score(newNegScoresX, priceChangeY))
    print('coeff ', reg3.coef_)
    print('intercept ', reg3.intercept_)
    
    reg4 = LinearRegression().fit(newTotalWordsX, priceChangeY)
    print('reg4 - newTotalWords')
    print('r squared ', reg4.score(newTotalWordsX, priceChangeY))
    print('coeff ', reg4.coef_)
    print('intercept ', reg4.intercept_)
    
    reg10 = LinearRegression().fit(newNonStemScoresX, priceChangeY)
    print('reg9 - newNonStemScores')
    print('r squared ', reg10.score(newNonStemScoresX, priceChangeY))
    print('coeff ', reg10.coef_)
    print('intercept ', reg10.intercept_)
    
    reg5 = LinearRegression().fit(twoFactors, priceChangeY)
    print('reg5 - wordScores, negWordScores')
    print('r squared ', reg5.score(twoFactors, priceChangeY))
    print('coeff ', reg5.coef_)
    print('intercept ', reg5.intercept_)
    
    #reg6 = LinearRegression().fit(threeFactors, priceChangeY)
    #print('reg6 - wordScores, negWordScores, newTotalWords')
    #print('r squared ', reg6.score(threeFactors, priceChangeY))
    #print('coeff ', reg6.coef_)
    #print('intercept ', reg6.intercept_)
    
    #print('threeFactors3 ', threeFactors3)
    reg9 = LinearRegression().fit(threeFactors3, priceChangeY)
    print('reg9 - wordScores, negWordScores, newNonStemScores')
    print('r squared ', reg9.score(threeFactors3, priceChangeY))
    print('coeff ', reg9.coef_)
    print('intercept ', reg9.intercept_)
    
    reg11 = LinearRegression().fit(newstembadx, priceChangeY)
    print('reg11 - newstembad')
    print('r squared', reg11.score(newstembadx, priceChangeY))
    print('coeff', reg11.coef_)
    
    reg12 = LinearRegression().fit(newstemnegatx, priceChangeY)
    print('reg 12 - newstemnegat')
    print('r squared', reg12.score(newstemnegatx, priceChangeY))
    print('coeff', reg12.coef_)
    
    reg13 = LinearRegression().fit(newstemgoodx, priceChangeY)
    print('reg 13 - newstemgood')
    print('r squared', reg13.score(newstemgoodx, priceChangeY))
    print('coeff', reg13.coef_)
    
    reg14 = LinearRegression().fit(newstemfailx, priceChangeY)
    print('reg 14 - newstemfail')
    print('r squared', reg14.score(newstemfailx, priceChangeY))
    print('coeff', reg14.coef_)
    
    reg15 = LinearRegression().fit(newstemterriblx, priceChangeY)
    print('reg 15 - newstemterribl')
    print('r squared', reg15.score(newstemterriblx, priceChangeY))
    print('coeff', reg15.coef_)
    
    reg16 = LinearRegression().fit(newstemunprofitx, priceChangeY)
    print('reg 16 - newstemunprofit')
    print('r squared', reg16.score(newstemunprofitx, priceChangeY))
    print('coeff', reg16.coef_)
    
    reg17 = LinearRegression().fit(newstempositx, priceChangeY)
    print('reg 17 - newstemposit')
    print('r squared', reg17.score(newstempositx, priceChangeY))
    print('coeff', reg17.coef_)
    
    reg18 = LinearRegression().fit(newstemgreatx, priceChangeY)
    print('reg 18 - newstemgreat')
    print('r squared', reg18.score(newstemgreatx, priceChangeY))
    print('coeff', reg18.coef_)
    
    reg19 = LinearRegression().fit(newstemsuccessx, priceChangeY)
    print('reg 19 - newstemsuccess')
    print('r squared', reg19.score(newstemsuccessx, priceChangeY))
    print('coeff', reg19.coef_)
    
    reg20 = LinearRegression().fit(newstemprofitx, priceChangeY)
    print('reg 20 - newstemprofit')
    print('r squared', reg20.score(newstemprofitx, priceChangeY))
    print('coeff', reg20.coef_)
    
    
    #IS curve data and plotting
    rsquaredReg1 = reg1.score(newScoresX, priceChangeY)
    rsquaredReg5 = reg5.score(twoFactors, priceChangeY)
    rsquaredReg9 = reg9.score(threeFactors3, priceChangeY)
    
    print('points ', list(zip([1,2,3], [rsquaredReg1,rsquaredReg5,rsquaredReg9])))
    
    plt.plot([1,2,3],[rsquaredReg1,rsquaredReg5,rsquaredReg9])
    plt.ylabel('r squared')
    plt.xlabel('number of factors')
    plt.show()
    
    
    
    

#determine which words are most negative/positive?
#highest correlation between word and future of stock price?
#need to code this
#combination of word scores is based on the individual word weights
#maybe one option could be 
#may go to outside sources  to get values of sentiment for words


#calculate in sample and out of sample regressions

#determine which words to use, as our factors
amznScores = getScores('AMZN')
regress_word(amznScores, 'AMZN')

#aaplScores = getScores('AAPL')
#regress_word(aaplScores, 'AAPL')

#spyScores = getScores('SPY')
#regress_word(spyScores, 'SPY')

#fbScores = getScores('FB')
#regress_word(fbScores, 'FB')

#iwmScores = getScores('IWM')
#regress_word(iwmScores, 'IWM')

#qqqScores = getScores('QQQ')
#regress_word(qqqScores, 'QQQ')

#bacScores = getScores('BAC')
#regress_word(bacScores, 'BAC')

#googlScores = getScores('GOOGL')
#regress_word(googlScores, 'GOOGL')

#jpmScores = getScores('JPM')
#regress_word(jpmScores, 'JPM')

#tslaScores = getScores('TSLA')
#regress_word(tslaScores, 'TSLA')









