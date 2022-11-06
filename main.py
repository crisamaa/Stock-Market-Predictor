import math
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
 
days = 10000
stock = '^GSPC'


start = (datetime.date.today() - datetime.timedelta( days ) )
end = datetime.datetime.today()

data = yf.download(stock, start=start, end=end, interval = '1d')
print(data.head())


plt.plot(data['Close'])
plt.show()