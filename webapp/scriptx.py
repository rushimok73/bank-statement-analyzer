# def functionx(x,y):
#     return x+y
# import pandas as pd
# import os
#
# def functionz():
#     path = open(os.path.dirname(os.path.realpath(__file__)) + '\p1data.csv', "r")
#     data = pd.read_csv(path)
#     data.pop("txnId")
#     print(data)

import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime
from statistics import mean
import requests
from bs4 import BeautifulSoup
from csv import writer
import os

def functiony(igoal,iyears):
    path = open(os.path.dirname(os.path.realpath(__file__)) + '\p1data.csv', "r", encoding='utf-8-sig')
    data = pd.read_csv(path)
    data.pop('txnId')
    data.pop('reference')

    data['narration'] = data['narration'].apply(lambda x:x.translate(str.maketrans('','','1234567890')))
    data['narration'] = data['narration'].apply(lambda x:re.sub(r"[,.;@#?!&$:_/-]+\ *"," ", x))
    def tolower(a):
        temp = []
        for i in a:
            temp.append(i.lower())
        return temp[1:]
    tokenizer=WordPunctTokenizer()
    data['nar_terms'] = data.apply(lambda row: tokenizer.tokenize(row.narration), axis = 1)
    data['nar_terms_without_mode'] = data.apply(lambda row: tolower(row['nar_terms']), axis = 1)

    def finddict(a):
        temp = []
        for i in a:
            term = str(i)
            if term in catedict:
                return catedict[term]
    catedict = {
    "amzn":"Amazon",
    "ola":"Ola",
    "swiggy":"Swiggy",
    "irctc":"IRCTC",
    "exidelife":"ExideLife",
    "cash":"Cash Withdrawal",
    "bse":"Bombay Stock Exchange",
    "motilal":"Motilal Oswal",
    "mtl":"Motilal Oswal",
    "modirect":"Mutual Funds",
    "lic":"LIC",
    "fintech": "Sal fintech products",
    "car":"Car Loan",
    "emi":"EMI",
    "easebuzz":"Easebuzz",
    "kotak":"Kotak Life Insurance",
    "rent":"Rental",
    "fashion": "Fashion",
    "zomato":"Zomato",
    "olacabs":"Ola",
    "easemytrip":"Ease My Trip",
    "mutual":"Mutual Funds",
    "nasdq":"NASDAQ",
    "dish":"Dish TV",
    "wwweasemytripco":"Ease My Trip",
    "pmsby":" Pradhan Mantri Suraksha Bima Yojana",
    }
    data['category'] = data.apply(lambda row: finddict(row['nar_terms_without_mode']), axis = 1)


    data = data.replace(to_replace='None', value=np.nan).dropna()



    data.groupby('category').amount.sum().sort_values().plot(kind='barh',figsize=(13,6))
    plt.ylabel('Category')
    plt.xlabel('Spending amount')
    plt.title('Spend by category')
    plt.savefig("testa.png")

    newdata = data[['valueDate','type','category']].copy()
    newdata['valueDate'] = pd.to_datetime(newdata['valueDate'], format = '%Y-%m-%d', errors='coerce')
    df1 = pd.DataFrame({"keys":[],"frequency":[]})
    keys = newdata.category.unique()
    j = 0
    for key in keys:
        amdates = []
        for label, row in newdata.iterrows():
            if row['category'] == str(key) and row['type'] == "DEBIT":
                amdates.append(row['valueDate'])
        if len(amdates) > 2:
            timedeltas = [amdates[i+1]-amdates[i] for i in range(len(amdates)-1)]
            average_timedelta = sum(timedeltas, datetime.timedelta(0)) / (len(amdates)-1)
            df1.loc[j] = key,average_timedelta
        j = j+1

    cutoff_lower = datetime.timedelta(
        days = 27
    )
    cutoff_upper = datetime.timedelta(
        days = 33
    )
    monthly=[]
    for label,row in df1.iterrows():
        if cutoff_upper >= row['frequency'] >= cutoff_lower:
            monthly.append(row['keys'])
    monthly_exp = []
    for key in monthly:
        exp = []
        for label,row in data.iterrows():
            if row['category'] == str(key):
                exp.append(row['amount'])
        monthly_exp.append(mean(exp))
    total_monthly_exp = sum(monthly_exp)



    response = requests.get('https://www.bajajfinserv.in/fixed-deposit-fees-and-interest-rates')

    soup = BeautifulSoup(response.text,'html.parser')

    posts = soup.find_all(class_="table table-bordered text-center")[0].find_all('tr')


    focus = posts[3].find_all('td')
    min_amt = focus[1].get_text()
    min_amt = min_amt.replace(',', '')
    rate = focus[2].get_text()
    rate = rate.replace('%', '')
    min_amt = float(min_amt)
    rate = float(rate)/100

    compounding_rate = 1
    # principal = ipricipal
    # time = iyears
    # if principal >= min_amt:
    #     maturity_amount = principal*((1+(rate/compounding_rate))**(time*compounding_rate))

    time = iyears
    goal = igoal
    principal = goal * ( (1 + (rate/compounding_rate)) ** (-1 * (time*compounding_rate) ) )
    return principal
