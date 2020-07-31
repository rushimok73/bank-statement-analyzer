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
import os,sys
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle

def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], \
          colors='jet_r', arrow=1, title='', fname=False):


    N = len(labels)

    if arrow > N:
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))



    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list):
        if len(colors) == N:
            colors = colors[::-1]
        else:
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))



    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]

    patches = []
    for ang, c in zip(ang_range, colors):
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

    [ax.add_patch(p) for p in patches]



    for mid, lab in zip(mid_points, labels):

        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=14, \
            fontweight='bold', rotation = rot_text(mid))


    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)

    ax.text(0, -0.05, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=22, fontweight='bold')



    pos = mid_points[abs(arrow - N)]

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))


    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig("gauge.png", dpi=200)
    if fname:
        fig.savefig("gauge.png", dpi=200)

def degree_range(n):
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points

def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation

def functiony(igoal,idate):
    #------------------Import Data-----------------------#
    path = open(os.path.dirname(os.path.realpath(__file__)) + '\p1data.csv', "r", encoding='utf-8-sig')
    data = pd.read_csv(path)
    data.pop('txnId')
    data.pop('reference')
    plt.clf()
    sdata = data
    #------------------Data Cleaning-----------------------#
    data['narration'] = data['narration'].apply(lambda x:x.translate(str.maketrans('','','1234567890')))
    data['narration'] = data['narration'].apply(lambda x:re.sub(r"[,.;@#?!&$:_/-]+\ *"," ", x))
    def tolower(a):
        temp = []
        for i in a:
            temp.append(i.lower())
        return temp[1:]
    #------------------Tokenize + Use dict-----------------------#
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
    "bse":"BSE",
    "motilal":"Motilal Oswal",
    "mtl":"Motilal Oswal",
    "modirect":"Mutual Funds",
    "lic":"LIC",
    "fintech": "Fintech Service",
    "car":"Car Loan",
    "emi":"EMI",
    "easebuzz":"Easebuzz",
    "kotak":"Kotak Life Ins.",
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
    #------------------Categories Graph-----------------------#
    data.groupby('category').amount.sum().sort_values().plot(kind='barh',figsize=(13,6))
    plt.ylabel('Category')
    plt.xlabel('Spending amount')
    plt.title('Spend by category')
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig("spending-by-category.png")
    plt.clf()
    #------------------Find frequencies of categories and filter for 30 days-----------------------#
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
    #------------------Scrape Data from Bajaj Finserv-----------------------#
    response = requests.get('https://www.bajajfinserv.in/fixed-deposit-fees-and-interest-rates')
    soup = BeautifulSoup(response.text,'html.parser')
    posts = soup.find_all(class_="table table-bordered text-center")[0].find_all('tr')
    focus = posts[3].find_all('td')
    min_amt = focus[1].get_text()
    min_amt = min_amt.replace(',', '')
    rate = focus[2].get_text()
    rate = rate.replace('%', '')
    min_amt = float(min_amt)
    rate = float(rate)
    #------------------Rushikesh Principal Calculations based on GOAL-----------------------#
    input_date = idate
    print(idate)
    goal_date = datetime.datetime.strptime(input_date, "%Y-%m-%d")
    current_time=datetime.datetime.now()
    print(goal_date)
    diff_time = (goal_date-current_time)/30
    months=int(diff_time.days)
    print(months)
    rate = rate/100
    compounding_rate = 1
    time = months
    goal = igoal
    fd_principal = goal * ( (1 + (rate/compounding_rate)) ** (-1 * (time*compounding_rate) ) )
    #------------------Suchit Graph 1----------------------#
    ttype = {}
    mtype = {}
    for label,row in sdata.iterrows():
        if row['type'] in ttype:
            ttype[row['type']] += row.amount
        else:
            ttype[row['type']] = row.amount
        if row['mode'] in mtype:
            mtype[row['mode']] += row.amount
        else:
            mtype[row['mode']] = row.amount
    keys = ttype.keys()
    values = ttype.values()
    plt.title('Total transactions per mode')
    bar1 = plt.bar(mtype.keys(), mtype.values())
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig('total-transactions-per-mode.png')
    plt.clf()
    #------------------Suchit Graph 2-----------------------#
    month = {}
    for label,row in sdata.iterrows():
        datee = datetime.datetime.strptime(row['valueDate'], "%Y-%m-%d")
        if str(datee.month)+"-"+str(datee.year) in month:
            month[str(datee.month)+"-"+str(datee.year)]+=row['amount']
        else:
            month[str(datee.month)+"-"+str(datee.year)]=row['amount']
    bar1 = plt.bar(month.keys(), month.values())
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.title('Monthly total transactions')
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig('monthly-total-transactions.png')
    plt.clf()
    #------------------Suchit Graph 3-----------------------#
    cdmonth = {}
    for label,row in sdata.iterrows():
        datee = datetime.datetime.strptime(row['valueDate'], "%Y-%m-%d")
        if row['type'] in cdmonth:
            if str(datee.month)+"-"+str(datee.year) in cdmonth[row['type']]:
                cdmonth[row['type']][str(datee.month)+"-"+str(datee.year)] += row.amount
            else:
                cdmonth[row['type']][str(datee.month)+"-"+str(datee.year)] = row.amount
        else:
            cdmonth[row['type']] = {}
            cdmonth[row['type']][str(datee.month)+"-"+str(datee.year)]=row['amount']

    da = pd.DataFrame(cdmonth)
    barWidth = 0.35
    da = pd.DataFrame(cdmonth)
    print(da)
    barDebit = da.iloc[:,0]
    barCredit = da.iloc[:,1]
    indx = np.arange(len(da))
    indx2 = [x + barWidth for x in indx]
    plt.figure(figsize=(10,7))

    graphDebit = plt.bar(x=indx,height=barDebit,width=barWidth, label = "Debit")
    graphCredit = plt.bar(x=indx2,height=barCredit,width=barWidth, label = "Credit")
    for rect in graphDebit:
        height=rect.get_height()
        plt.text(rect.get_x()-0.05 + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

    for rect in graphCredit:
        height=rect.get_height()
        plt.text(rect.get_x()+0.1 + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

    plt.xlabel('Values')
    plt.ylabel('Amount')
    plt.title('Monthly debit and credit')
    plt.xticks(range(0,len(cdmonth['DEBIT'])),cdmonth['DEBIT'].keys())
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig('monthly_debit_credit.png')
    plt.clf()
    #------------------Suchit Graph 4-----------------------#
    cdtmonth = {}
    for label,row in sdata.iterrows():
        datee = datetime.datetime.strptime(row['valueDate'], "%Y-%m-%d")
        if row['mode'] in cdtmonth:
            if str(datee.month)+"-"+str(datee.year) in cdtmonth[row['mode']]:
                cdtmonth[row['mode']][str(datee.month)+"-"+str(datee.year)] += row.amount
            else:
                cdtmonth[row['mode']][str(datee.month)+"-"+str(datee.year)] = row.amount
        else:
            cdtmonth[row['mode']] = {}
            cdtmonth[row['mode']][str(datee.month)+"-"+str(datee.year)]=row['amount']
    da = pd.DataFrame(cdtmonth)
    da.fillna(0,inplace = True)
    barUPI = da.iloc[:,0]
    barOther = da.iloc[:,1]
    barCard = da.iloc[:,2]
    barATM = da.iloc[:,3]
    barFT = da.iloc[:,4]
    barCash = da.iloc[:,5]
    indx = np.arange(len(da))
    plt.figure(figsize=(10,7))
    graphUPI = plt.bar(x=indx,height=barUPI,width=0.35, label = "UPI")
    graphOther = plt.bar(x=indx,height=barOther,width=0.35,bottom=barUPI, label = "Other")
    graphCard = plt.bar(x=indx,height=barCard,width=0.35,bottom=barOther, label = "Card")
    graphATM = plt.bar(x=indx,height=barATM,width=0.35,bottom=barCard, label = "ATM")
    graphFT = plt.bar(x=indx,height=barFT,width=0.35,bottom=barATM,label = "Fund Transfer")
    graphCash = plt.bar(x=indx,height=barCash,width=0.35,bottom=barFT, label = "Cash")
    plt.title('Modes of transactions monthly')
    plt.legend()
    plt.xlabel('Values')
    plt.ylabel('Amount')
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig('transaction-modes-monthly.png')
    plt.clf()
    #------------------Suchit Graph 5-----------------------#
    savings = {}
    sav = 0
    for label,row in sdata.iterrows():
        datee = datetime.datetime.strptime(row['valueDate'], "%Y-%m-%d")
        if str(datee.month)+"-"+str(datee.year) in savings:
            if row['type'] == 'DEBIT':
                savings[str(datee.month)+"-"+str(datee.year)] -= row.amount
            else:
                savings[str(datee.month)+"-"+str(datee.year)] += row.amount
        else:
            if(not not savings):
                sav += list(savings.values())[-1]
            savings[str(datee.month)+"-"+str(datee.year)] = 0
            if row['type'] == 'DEBIT':
                savings[str(datee.month)+"-"+str(datee.year)] -= row.amount
            else:
                savings[str(datee.month)+"-"+str(datee.year)] += row.amount
    sav += list(savings.values())[-1]

    colr = ['#d63031' if (x < 0) else '#00b894' for x in savings.values()]
    bar1 = plt.bar(savings.keys(),savings.values(),color=colr)

    for rect in bar1:
        height = rect.get_height()
        if height<0:
            plt.text(rect.get_x() + rect.get_width()/2.0, 0, '%d' % int(height), ha='center', va='bottom')
        else:
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.axhline(y=0,color="black")

    tot_savings = 0
    for key in savings:
        tot_savings = tot_savings + savings[key]
    print(tot_savings)
    avg_savings = tot_savings/len(savings)
    plt.axhline(y=avg_savings,linestyle = '--', color = '#6c5ce7',label = "Average Saving")
    plt.legend()
    plt.title('Monthly Savings')
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig('monthy-savings.png')
    plt.clf()
    #------------------Suchit Graph 6-----------------------#
    from sklearn import datasets, linear_model
    cdmonth = {}
    for label,row in sdata.iterrows():
        datee = datetime.datetime.strptime(row['valueDate'], "%Y-%m-%d")
        if str(datee.month)+"-"+str(datee.year) in cdmonth:
            if row['type'] in cdmonth[str(datee.month)+"-"+str(datee.year)]:
                cdmonth[str(datee.month)+"-"+str(datee.year)][row['type']] += row.amount
            else:
                cdmonth[str(datee.month)+"-"+str(datee.year)][row['type']] = row.amount
        else:
            cdmonth[str(datee.month)+"-"+str(datee.year)] = {}
            cdmonth[str(datee.month)+"-"+str(datee.year)][row['type']]=row['amount']
    x = []
    for month in cdmonth:
        x.append(cdmonth[month]['DEBIT'])
    y = list(range(0,len(x)))
    x= np.array(x).reshape(len(x),1)
    y= np.array(y).reshape(len(y),1)
    yt = list(range(0,len(x)+5))
    yt= np.array(yt).reshape(len(yt),1)
    regr = linear_model.LinearRegression()
    regr.fit(y,x)
    plt.scatter(y, x,  color='black')
    plt.plot(yt, regr.predict(yt), color='red',linewidth=3)
    plt.title('Savings predictions MoM')
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig('savings_predictions_MoM.png')
    plt.clf()
    #------------------Suchit Graph 7-----------------------#
    # goal = igoal
    # months = imonths
    # r = rate
    # P=goal/(months-1)
    # i=r/(12*100)
    # n=months
    # y = []
    ax = plt.subplots()
    goal = igoal
    r = rate*100
    P=goal/(months-1)
    i=r/(12*100)
    n=months
    y = []

    for m in range(2,n+2):
        M= P*((1+i)**m-1)/i-P
        if((m-1)%3==0):
            bar1 = plt.bar(x=m-1,height=M,color="#9c88ff")
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
            y.append(P*(m-1)-P)
    plt.bar(x=0,height=0,label="Amount after investment",color="#9c88ff")
    x = range(3,n+2,3)
    plt.plot(x,y,color='black',label="Amount without investment")
    plt.legend()
    plt.xticks(x)
    plt.title('Savings vs RD')
    plt.xlabel("Months")
    plt.ylabel("Amount")
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.tight_layout()
    plt.savefig('savings-vs-rd.png')
    plt.clf()
    if P <= avg_savings:
        purchase_possible = "Yes"
    else:
        purchase_possible = "No"
    profit = M-goal
    #------------------GST_Data---------------------------#
    path = open(os.path.dirname(os.path.realpath(__file__)) + '\gstdata.csv', "r", encoding='utf-8-sig')
    data = pd.read_csv(path)
    data.pop('transactionTimestamp')
    data.pop('reference')
    data['narration'] = data['narration'].apply(lambda x:re.sub(r"[,.;@#?!&$:_/-]+\ *"," ", x))
    data['nar_terms'] = data.apply(lambda row: tokenizer.tokenize(row.narration), axis = 1)
    data['category'] = data.apply(lambda row: row.nar_terms[0], axis = 1)
    data['product'] = data.apply(lambda row: row.nar_terms[1], axis = 1)
    keys = data['product'].unique()
    df1 = pd.DataFrame({"prodkey":[],"SOGdate":[],"GSTdate":[]})
    i=0
    for key in keys:
        status = -1
        for label, row in data.iterrows():
            if row['product'] == str(key) and row['category'] == "SOG":
                status = 0
                sogdate = row['valueDate']
            if row['product'] == str(key) and row['category'] == "GSTIN" and status == 0:
                status = 1
                df1.loc[i] = key, sogdate, row['valueDate']
                i = i+1
                break
    df1['SOGdate'] = pd.to_datetime(df1['SOGdate'], format = '%d-%m-%Y', errors='coerce')
    df1['GSTdate'] = pd.to_datetime(df1['GSTdate'], format = '%d-%m-%Y', errors='coerce')
    for label, row in df1.iterrows():
        df1['difference'] = df1['GSTdate'] - df1['SOGdate']
    score = 5
    for label, row in df1.iterrows():
        if datetime.timedelta(days = 0)<=row['difference'] and row['difference']<=datetime.timedelta(days = 50):
            score = score + 1
        elif datetime.timedelta(days = 50)<row['difference'] and row['difference']<=datetime.timedelta(days = 80):
            score = score - 0.25
        elif datetime.timedelta(days = 80)<row['difference'] and row['difference']<=datetime.timedelta(days = 110):
            score = score - 0.50
        elif datetime.timedelta(days = 110)<row['difference'] and row['difference']<=datetime.timedelta(days = 140):
            score = score - 0.75
        elif datetime.timedelta(days = 140)<row['difference']:
            score = score - 1
    #------------------GST_Score_graph---------------------------#
    arr = int(score)


    gauge(labels=['','','','','','','','','',''],colors='RdYlGn_r', arrow=arr, title='GST  Score = '+str(score))
    plt.savefig("gaugex.png")

    return P,avg_savings,profit,purchase_possible,fd_principal,score
