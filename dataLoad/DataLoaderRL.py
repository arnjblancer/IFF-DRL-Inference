import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from operator import itemgetter
from dataLoad.utils import preProcessor

class RLDataSet(Dataset):

    def __init__(self, csvFilePath, startDate, endDate, cfg
                 # normalization
                 ):
        self.datas = []
        self.PriceTMinus1 = []
        self.PriceT = []
        self.PriceTPlusN = []
        self.mask = []
        self.TPlusN = cfg.rewardDay
        self.predict_datas = []
        self.day_values = []
        self.week_values = []
        self.start_point = cfg.input.start_point
        _,_, data, self.data_stamp = preProcessor(pd.read_csv(csvFilePath), startDate, endDate)
        _data, _priceTMinus1, _priceToday, _PriceTPlusN, _mask = self.splitData(data,self.data_stamp, cfg.input.length, cfg.output.length)
        self.datas += _data
        self.PriceTMinus1 += _priceTMinus1
        self.PriceT += _priceToday
        self.PriceTPlusN += _PriceTPlusN
        self.mask += _mask
        self.length = len(self.datas)

    def splitData(self, data,mask, length, predictLength):
        _datas = []
        _predict_data = []
        _masks = []
        _priceTPlusN = []
        _priceTMinus1 = []
        _priceToday = []
        _day_value = []
        _week_value = []

        for i in range(len(data) - length - self.TPlusN + 1):
            _masks.append(mask[i: i + length])
            _priceTMinus1.append(data[i + length - 2, 3])
            _priceToday.append(data[i + length - 1, 3])
            _priceTPlusN.append(data[i + length: i + length + self.TPlusN, 3])
            _datas.append(data[i: i + length ,:])
        return _datas, _priceTMinus1, _priceToday, _priceTPlusN, _masks

    def __len__(self):
        return self.length-1

    def __getitem__(self, index):
        return self.datas[index], self.PriceTMinus1[index], self.PriceT[index], self.PriceTPlusN[index], self.datas[index+1], self.mask[index], self.mask[index+1]  #self.predict_datas[index],self.predict_datas[index+1], self.day_values[index], self.week_values[index]

def stock_fn(batch):
    return [torch.from_numpy(np.array(list(map(itemgetter(i), batch)))).type(torch.FloatTensor) for i in range(len(batch[0]))]
