import math
import pandas as pd
import numpy as     
import yfinance as yf

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
print(sp500)