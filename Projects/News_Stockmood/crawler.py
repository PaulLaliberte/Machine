"""scraping morningstar data"""

import urllib2
import re
import csv

import bs4 as bs      #beautiful soup
import pandas as pd
from urllib2 import Request
from re import match

def readin_csv():
    tickers = []
    with open('companylist.csv', 'rb') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            tickers.append(row[0])
    return tickers

class crawler:

    def __init__(self, ticker_list=None):
        #(javascript_bool, url)
        self.stock_ticker = ticker_list
        self.df = None

    def crawl(self):
        numbers = ['-','0','1','2','3','4','5','6','7',
                    '8','9']

        half_1 = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?t='
        half_2 = '&reportType=is&period=12&dataType=A&order=asc&columnYear=5&number=3'

        for ticker in self.stock_ticker:
            url = half_1 + ticker + half_2
            req = Request(url)
            res = urllib2.urlopen(req)

            soup = bs.BeautifulSoup(res, 'lxml')
            soup = str(soup).split("\n")
            soup = soup[2:]

            signal_ = []
            signal = []
            data = []

            for i in soup:
                for j in range(0, len(i)):
                    if i[j] in numbers:
                        signal_.append(i[:j-1])
                        data.append(i[j:])
                        break

            punc = re.compile(r'["!?,.:;]')

            for i in signal_:
                index = punc.sub('', i)
                signal.append(index)

            df_signal = pd.DataFrame(signal)
            df_data = pd.DataFrame(data)
            self.df = pd.concat([df_signal, df_data], axis=1, ignore_index=True)

            self.df.to_pickle(str(ticker))


if __name__ == '__main__':
    list_of_tickers = readin_csv()
    n = crawler(list_of_tickers)
    n.crawl()

