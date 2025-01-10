import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

class PerformanceEstimator:

    def __init__(self, cfg, tradingData, saveDirName):
        self.saveDirName = saveDirName
        self.data = tradingData
        self.window_size = cfg.input.length

    def computePnL(self):
        self.PnL = self.data.loc[len(self.data) - 1, "Capitals"] - self.data.loc[0, "Capitals"]
        return self.PnL

    def computeCR(self):
        self.CR = (self.data.loc[len(self.data) - 1, "Capitals"] - self.data.loc[0, "Capitals"])/self.data.loc[0, "Capitals"]
        return self.CR
    def computeAnnualizedReturn(self, data_length):
        # Compute the cumulative return over the entire trading horizon
        cumulativeReturn = self.data['Returns'].cumsum()
        cumulativeReturn = cumulativeReturn.iloc[-1]

        timeElapsed = data_length-self.window_size
        print(cumulativeReturn)
        # Compute the Annualized Return
        if (cumulativeReturn > -1):
            self.annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365 / timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100

        return self.annualizedReturn

    def computeAnnualizedVolatility(self):

        # Compute the Annualized Volatility (252 trading days in 1 trading year)
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily

    def computeSharpeRatio(self, riskFreeRate=0):

        # Compute the expected return
        expectedReturn = self.data['Returns'].mean()

        # Compute the returns volatility
        volatility = self.data['Returns'].std()

        # Compute the Sharpe Ratio (252 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate) / volatility
        else:
            self.sharpeRatio = 0
        return self.sharpeRatio

    def computeSortinoRatio(self, riskFreeRate=0):

        # Compute the expected return
        expectedReturn = np.mean(self.data['Returns'])

        # Compute the negative returns volatility
        negativeReturns = [returns for returns in self.data['Returns'] if returns < 0]
        volatility = np.std(negativeReturns)

        # Compute the Sortino Ratio (252 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            self.sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate) / volatility
        else:
            self.sortinoRatio = 0
        return self.sortinoRatio

    def computeMaxDrawdown(self, plotting=False):

        # Compute both the Maximum Drawdown and Maximum Drawdown Duration
        capital = self.data['Capitals'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through != 0:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through]) / capital[peak]
            self.maxDDD = through - peak
        else:
            self.maxDD = 0
            self.maxDDD = 0
            return self.maxDD, self.maxDDD

        # Plotting of the Maximum Drawdown if required
        if plotting:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data['Capitals'], lw=2, color='Blue')
            plt.plot([self.data.iloc[[peak]].index, self.data.iloc[[through]].index],
                     [capital[peak], capital[through]], 'o', color='Red', markersize=5)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.savefig(''.join(['Figures/', 'MaximumDrawDown', '.png']))
            # plt.show()

        # Return of the results
        return self.maxDD, self.maxDDD

    def computeProfitability(self):

        # Initialization of some variables
        good = 0
        bad = 0
        profit = 0
        loss = 0
        index = next((i for i in range(len(self.data.index)) if self.data['Action'][i] != 0), None)
        if index == None:
            self.profitability = 0
            self.averageProfitLossRatio = 0
            return self.profitability, self.averageProfitLossRatio
        money = self.data['Capitals'][index]

        # Monitor the success of each trade over the entire trading horizon
        for i in range(index + 1, len(self.data.index)):
            if (self.data['Action'][i] == 2):
                delta = self.data['Capitals'][i] - money
                money = self.data['Capitals'][i]
                if (delta >= 0):
                    good += 1
                    profit += delta
                else:
                    bad += 1
                    loss -= delta

        # Special case of the termination trade
        delta = self.data.iloc[-1]['Capitals'] - money
        if (delta >= 0):
            good += 1
            profit += delta
        else:
            bad += 1
            loss -= delta

        # Compute the Profitability
        self.profitability = 100 * good / (good + bad)

        # Compute the ratio average Profit/Loss
        if (good != 0):
            profit /= good
        if (bad != 0):
            loss /= bad
        if (loss != 0):
            self.averageProfitLossRatio = profit / loss
        else:
            self.averageProfitLossRatio = float('Inf')

        return self.profitability, self.averageProfitLossRatio

    def computeSkewness(self):

        # Compute the Skewness of the returns
        self.skewness = self.data["Returns"].skew()
        return self.skewness

    def computePerformance(self, data_length):

        # Compute the entire set of performance indicators
        self.computePnL()
        self.computeCR()
        self.computeAnnualizedReturn( data_length)
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        self.computeSkewness()


        # Generate the performance table
        self.performanceTable = [["Profit & Loss (P&L)", "{0:.0f}".format(self.PnL)],
                                 ["Cumulative Return", "{0:.3f}%".format(self.CR*100)],
                                 ["Annualized Return", "{0:.2f}".format(self.annualizedReturn) + '%'],
                                 ["Annualized Volatility", "{0:.2f}".format(self.annualizedVolatily) + '%'],
                                 ["Sharpe Ratio", "{0:.3f}".format(self.sharpeRatio)],
                                 ["Sortino Ratio", "{0:.3f}".format(self.sortinoRatio)],
                                 ["Maximum Drawdown", "{0:.2f}".format(self.maxDD) + '%'],
                                 ["Maximum Drawdown Duration", "{0:.0f}".format(self.maxDDD) + ' days'],
                                 ["Profitability", "{0:.2f}".format(self.profitability) + '%'],
                                 ["Ratio Average Profit/Loss", "{0:.3f}".format(self.averageProfitLossRatio)],
                                 ["Skewness", "{0:.3f}".format(self.skewness)]]

        return self.performanceTable

    def as_csv(self, table, estimateSavePath):

        with open(estimateSavePath, "w") as f:
            writer = csv.writer(f)
            for line in table.split("\n"):
                writer.writerow(line.split())

    def test(self,table, headers):

        # 计算图片大小
        font = ImageFont.truetype("arial.ttf", 14)
        max_width = max(font.getsize(row)[0] for row in table.split('\n'))
        width, height = font.getsize(table.split('\n')[0])
        height *= len(table.split('\n')) + 5  # 增加5行的高度

        # 创建图片对象并绘制表格
        img = Image.new('RGB', (max_width, height), color='white')
        d = ImageDraw.Draw(img)
        d.text((0, 0), table, font=font, fill=(0, 0, 0))

        # 保存图片
        img.save('output.png')

    def as_image(self, table, estimateSavePath):

        font = ImageFont.load_default()
        max_width = max(font.getsize(row)[0] for row in table.split('\n'))
        width, height = font.getsize(table.split('\n')[0])
        height *= len(table.split('\n')) + 5  # 增加5行的高度

        # 创建图片对象并绘制表格
        img = Image.new('RGB', (max_width, height), color='white')
        d = ImageDraw.Draw(img)
        d.text((0, 0), table, font=font, fill=(0, 0, 0))

        # 保存图片
        img.save(estimateSavePath)

    def as_txt(self, table, headers, estimateSavePath):
        table_str = tabulate(table, headers=headers)

        with open(estimateSavePath, "w") as f:
            f.write(table_str)

    def displayPerformance(self, name, data_length):

        # Generation of the performance table
        self.computePerformance(data_length)

        # Display the table in the console (Tabulate for the beauty of the print operation)
        headers = ["Performance Indicator", name]
        tabulation = tabulate(self.performanceTable, headers, stralign="center")
        estimateSavePath = os.path.join(self.saveDirName, name+'_best_result_estimation.jpg')
        self.as_image(tabulation, estimateSavePath)
