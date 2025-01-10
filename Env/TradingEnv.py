import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from Model.TimesNet import TimesNet
# from algorithm.DataProcessor import Dataprocessor


class TradingEnv:
    def __init__(self, cfg, init_money=500000):
        self.init_money = init_money
        self.cfg =cfg
        self.t = 0  # 时刻
        self.yesterdayPrice = 0
        self.todayPrice = 0
        self.tomorrowPrice = 0
        self.futurePrice = None
        self.show_log = False
        self.length = 0
        self.RewardLength = self.cfg.agent.RewardLength
        self.initPrice = 0
        self.diff = 0
        self.const = 1.5
        self.price_change = 0
        self.dayCount = 0
        self.isFirstDay = True
        self.zeroDay = None
        self.Nday = None
        self.expert_action = None
        self.buyCount = 0
        self.sellCount= 0
        self.holdCount = 0

    def setInitPrice(self, price):
        self.initPrice = price

    def getCapitals(self):
        return self.account.iloc[-1]['Capitals']

    def get_Sharpe(self, action):
        chg = []
        for i in self.futurePrice[0, 0: self.RewardLength]:
            chg.append((i - self.todayPrice) / self.todayPrice)
        chg = torch.concat(chg)
        reward = (torch.mean(chg) / torch.std(chg))
        return action*reward

    def get_Min_Max(self, action):
        if action == 0:
            reward = self.const - self.diff
        else:
            reward = action*self.price_change
        return reward

    def get_Return(self, action):
        reward = ((self.futurePrice[0, self.RewardLength - 1] - self.todayPrice) / self.todayPrice) * 100
        
        return action*reward


    def getReward(self, type, action):
        if type == 'sharpe':
            reward = self.get_Sharpe(action)
        elif type == 'return':
            reward = self.get_Return(action)
        elif type == 'MinMax':
            reward = self.get_Min_Max(action)
        else:
            print('Wrong type of reward!')
            reward = 0
        return reward


    def getNday(self):
        positive_ratio = []
        negative_ratio = []
        return_ratio = []
        first = torch.tensor(0.0).to(self.cfg.device)
        return_ratio.append(first)

        for i in self.futurePrice[0, 0: 3]:
            ratio = ((i - self.todayPrice) / self.todayPrice)*100
            return_ratio.append(ratio)


        max_ratio, min_ratio = max(return_ratio), min(return_ratio)
        if min_ratio>0:
            self.price_change = max_ratio
        elif max_ratio<0:
            self.price_change = min_ratio
        else:
            self.price_change = max_ratio if max_ratio+min_ratio > 0 else min_ratio

        self.diff = max_ratio - min_ratio
        return self.price_change, self.diff




    def setState(self, t, yesterdayPrice, todayPrice, futurePrice):
        self.t = t+1
        self.account.loc[self.t, 'Close'] = todayPrice.cpu().numpy()
        self.yesterdayPrice = yesterdayPrice
        self.todayPrice = todayPrice
        self.tomorrowPrice = futurePrice[0,0]
        self.futurePrice = futurePrice

    def buy(self):
        daily_chg = 0
        if len(self.profit_rate_account) == 0:
            last_value = 0
        else:
            last_value = self.profit_rate_account[-1]

        if self.account.loc[self.t - 1, 'Position'] == -1:  # 有空单 -> 平仓
            daily_chg = (self.yesterdayPrice - self.todayPrice) / self.yesterdayPrice
            profit_rate = (last_value + 1) * daily_chg + last_value
            self.states_close.append(self.t)
            self.short_pos = False
            self.account.loc[self.t, 'Cash'] = self.account.loc[
                                                              self.t - 1, 'Cash'] - self.holdingNum * \
                                                          self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)
            self.account.loc[self.t, 'Position'] = 0
            self.account.loc[self.t, 'Action'] = 2
            self.holdingNum = 0
            self.trade_profit.append(self.last_opening_price - self.todayPrice)

            if self.show_log:
                print('day:%d, close_short_pos price:%f' % (self.t, self.todayPrice))
        elif self.account.loc[self.t - 1, 'Position'] == 0:  # 空仓 -> 多单
            self.states_buy.append(self.t)
            # 计算买股量
            self.holdingNum = math.floor(
                self.account.loc[self.t - 1, 'Cash'] / (
                            self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)))
            # 计算现金变动
            self.account.loc[self.t, 'Cash'] = self.account.loc[
                                                              self.t - 1, 'Cash'] - self.holdingNum * \
                                                          self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)
            self.account.loc[self.t, 'Position'] = 1
            self.account.loc[self.t, 'Action'] = 1
            profit_rate = last_value
            self.last_opening_price = self.todayPrice

            if self.show_log:
                print('day:%d, long_pos price:%f' % (self.t, self.todayPrice))
        else:  # 无空单且有多单不操作
            self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash']
            self.account.loc[self.t, 'Position'] = self.account.loc[self.t - 1, 'Position']
            self.account.loc[self.t, 'Action'] = 0
            daily_chg = (self.todayPrice - self.yesterdayPrice) / self.yesterdayPrice
            profit_rate = (last_value + 1) * daily_chg + last_value
        self.daily_return .append(daily_chg)
        self.profit_rate_account.append(profit_rate)

    def sell(self):
        daily_chg = 0
        if len(self.profit_rate_account) == 0:
            last_value = 0
        else:
            last_value = self.profit_rate_account[-1]
        t = self.t
        if self.account.loc[t - 1, 'Position'] == 1:  # 有多单 -> 无持仓
            daily_chg = (self.todayPrice - self.yesterdayPrice) / self.yesterdayPrice
            profit_rate = (last_value + 1) * daily_chg + last_value
            self.states_close.append(self.t)
            self.account.loc[t, 'Cash'] = self.account.loc[
                                                         t - 1, 'Cash'] + self.holdingNum * \
                                                     self.account.loc[t, 'Close'] * (1 - self.transitionCost)
            self.holdingNum = 0
            self.account.loc[t, 'Position'] = 0
            self.account.loc[t, 'Action'] = 2
            self.trade_profit.append(self.todayPrice - self.last_opening_price)

            if self.show_log:
                print('day:%d, close_long_pos price:%f' % (self.t, self.todayPrice))

        elif self.account.loc[t - 1, 'Position'] == 0:  # 无多单且无空单就开空单
            self.states_sell.append(self.t)
            self.holdingNum = math.floor(
                self.account.loc[t - 1, 'Cash'] / (self.account.loc[t, 'Close'] * (1 + self.transitionCost)))
            self.account.loc[t, 'Cash'] = self.account.loc[
                                                         t - 1, 'Cash'] + self.holdingNum * \
                                                     self.account.loc[t, 'Close'] * (1 - self.transitionCost)
            self.account.loc[t, 'Action'] = -1
            self.account.loc[t, 'Position'] = -1
            profit_rate = last_value
            self.last_opening_price = self.todayPrice

            if self.show_log:
                print('day:%d, short_pos price:%f' % (self.t, self.todayPrice))
        else:  # 无多单且有空单不操作
            self.account.loc[t, 'Cash'] = self.account.loc[t - 1, 'Cash']
            self.account.loc[t, 'Action'] = 0
            self.account.loc[t, 'Position'] = self.account.loc[t - 1, 'Position']
            daily_chg = (self.yesterdayPrice - self.todayPrice) / self.yesterdayPrice
            profit_rate = (last_value + 1) * daily_chg + last_value


        self.daily_return .append(daily_chg)
        self.profit_rate_account.append(profit_rate)

    def riskControl(self):
        if len(self.profit_rate_account) == 0:
            last_value = 0
        else:
            last_value = self.profit_rate_account[-1]
        if self.account.loc[self.t - 1, 'Position'] == 1:
            profit_rate = (self.todayPrice - self.yesterdayPrice) / self.yesterdayPrice
            profit_rate = (last_value + 1) * profit_rate + last_value
        elif self.account.loc[self.t - 1, 'Position'] == -1:
            profit_rate = (self.yesterdayPrice - self.todayPrice) / self.yesterdayPrice
            profit_rate = (last_value + 1) * profit_rate + last_value
        else:
            profit_rate = last_value

        self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash']
        self.account.loc[self.t, 'Action'] = 0
        self.account.loc[self.t, 'Position'] = self.account.loc[self.t - 1, 'Position']
        self.profit_rate_account.append(profit_rate)

    def agentStep(self, state, action, position=False):
        if action==2:
            action=torch.tensor(-1).to(self.cfg.device)
        if action == 1:
            self.account.loc[self.t, 'Q_Action'] = 1
            self.buy()
            self.buyCount += 1
        elif action == -1:
            self.account.loc[self.t, 'Q_Action'] = -1
            self.sell()
            self.sellCount += 1
        elif action == 0:  # 不操作
            self.account.loc[self.t, 'Q_Action'] = 0
            self.riskControl()
            self.holdCount += 1

        reward = self.getReward(self.cfg.train.reward, action)

        self.account.loc[self.t, 'Reward'] = reward.cpu().numpy()
        self.account.loc[self.t, 'Holdings'] = self.account.loc[self.t, 'Position'] * self.holdingNum * self.account.loc[self.t, 'Close']
        self.account.loc[self.t, 'Capitals'] = self.account.loc[self.t, 'Cash'] + self.account.loc[self.t, 'Holdings']
        self.account.loc[self.t, 'Returns'] = (self.account.loc[self.t, 'Capitals'] -self.account.loc[self.t - 1, 'Capitals']) / \
                                                         self.account.loc[self.t - 1, 'Capitals']
        profit_rate_stock = (self.todayPrice - self.initPrice) / self.initPrice
        self.profit_rate_stock.append(profit_rate_stock)
        return reward.squeeze()

    def resetPandasTable(self):
        self.account = pd.DataFrame(
            columns=[],
            index=np.arange(0, self.length+1),
        )
        self.account['close'] = 0

        self.account['Position'] = 0  # 0 : 空仓  1 : 多单  -1 : 空单
        self.account['Action'] = 0  # 0 : 持有  1 : 开多单  -1 : 开空单  2 : 平仓
        self.account['Q_Action'] = 0  # 网络给出的action， 0 : hold  1 : buy  -1 : sell
        self.account['Holdings'] = 0.0  # 持仓市值
        self.account['Cash'] = float(self.init_money)  # 现金
        self.account['Capitals'] = self.account['Holdings'] + self.account[
            'Cash']  # 总市值
        self.account['Returns'] = 0.0  # 每日盈亏
        self.account['NDC'] = 0  # 下一天波动作为reward
        self.account['N5SR'] = 0  # 后五天的SR作为reward

    def reset(self, length):
        # self.t = np.clip(startingPoint, self.window_size - 1, self.terminal_date - self.window_size)
        self.length = length
        self.resetPandasTable()
        # a = pd.date_range(datetime.strptime(self.startingDate, '%Y-%m-%d'), datetime.strptime(self.endingDate, '%Y-%m-%d'), freq='D')
        self.margin = 0.0#每日保证金费用
        self.borrowing = 0.0#每日借款费用
        self.holdingNum = 0  # 持股量
        self.transitionCost = 0.003 # previous:0.0001
        self.borrowingFee = 0.02 #年借款费率
        self.marginCost = 0.005 #保证金要求率
        self.marginCost_annual = 0.03#保证金年利率
        self.long_pos = False  # 是否持有多单
        self.short_pos = False  # 是否持有空单
        self.last_value = self.init_money  # 上一天市值
        self.reward = 0  # 收益
        self.states_sell = []  # 卖股票时间
        self.states_buy = []  # 买股票时间
        self.states_close = []  # 平仓时间
        self.profit_rate_account = []  # 账号盈利
        self.profit_rate_stock = []  # 股票波动情况
        self.daily_return  = []  # 每日盈亏比例
        self.winning_trades  = 0
        self.total_trades = 0
        self.last_opening_price = 0  # 开仓价
        self.trade_profit = []  # 单笔盈亏
