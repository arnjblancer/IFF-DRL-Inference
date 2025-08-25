import math
import os

import numpy
import numpy as np
import pandas as pd
import torch
from torch import exp
from dataLoad.timefeatures import time_features

def preProcessor(data, start_date='2007-01-01', end_date='2017-12-31'):
    """
    :param data: csv read data
    :param start_date: date like 2007-01-01
    :param end_date: date like 2017-12-31
    :return: numpy
    """
    data = data.dropna()
    data = data[-data.Volume.isin([0])]

    if len(data.columns)==7:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        df_stamp = data[['Date']]
        data_stamp = time_features(df_stamp, timeenc=1, freq='h')
        #data = data.drop(['Close'], axis=1)
        data.reset_index(drop=True, inplace=True)
        index = data.index
        price_data = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
        return index, data, np.array(price_data), np.array(data_stamp)

    elif len(data.columns)==12:
        input_data = data.dropna()
        input_data['Date'] = pd.to_datetime(input_data['Date'])
        input_data = input_data.loc[(input_data['Date'] >= start_date) & (input_data['Date'] <= end_date)]
        input_data = input_data[-input_data.Volume.isin([0])]
        # Drop duplicated weekly columns and the unused Original Price column so that
        # only daily OHLCV features remain. This matches the expected five-channel
        # input of the model.
        input_data = input_data.drop(
            ['Open.1', 'High.1', 'Low.1', 'Adj Close.1', 'Volume.1', 'Original Price'],
            axis=1,
        )
        df_stamp = input_data[['Date']]
        data_stamp = time_features(df_stamp, timeenc=1, freq='h')
        return_data = input_data
        input_data = input_data.drop(['Date'], axis=1)

        input_data.reset_index(drop=True, inplace=True)
        return None, return_data, np.array(input_data), np.array(data_stamp)



def get_date(datasetName, start_date='2007-01-01', end_date='2017-12-31'):
    data = pd.read_csv('Incremental_Data/{}.csv'.format(datasetName))
    data = data.dropna()
    data = data[-data.Volume.isin([0])]
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    data.reset_index(drop=True, inplace=True)
    return data['Date']



def normalization(x):
    batch, length, channel = x.shape
    if channel==10:
        day_volume = x[:, :, 4]
        day_volume = (day_volume - torch.mean(day_volume)) / (torch.std(day_volume) + 0.0001)
        week_volume = x[:, :, 9]
        week_volume = (week_volume - torch.mean(week_volume)) / (torch.std(week_volume) + 0.0001)
        x[:, :, 4] = day_volume
        x[:, :, 9] = week_volume
        day_data = x[:, :, :4]
        day_data = (day_data - torch.mean(day_data)) / (torch.std(day_data) + 0.0001)
        week_data = x[:, :, 5:9]
        week_data = (week_data - torch.mean(week_data)) / (torch.std(week_data) + 0.0001)
        x[:, :, :4] = day_data
        x[:, :, 5:9] = week_data
    elif channel==5:
        volume = x[:, :, 4]
        volume = (volume - torch.mean(volume)) / (torch.std(volume) + 0.0001)
        x[:, :, 4] = volume
        data = x[:, :, :4]
        data = (data - torch.mean(data)) / (torch.std(data) + 0.0001)
        x[:, :, :4] = data
    elif channel==8:
        day = x[:,:,:4]
        week = x[:,:,4:8]
        day = (day - torch.mean(day)) / (torch.std(day) + 0.0001)
        week = (week - torch.mean(week)) / (torch.std(week) + 0.0001)
        x[:,:,:4] = day
        x[:,:,4:8] = week

    else:
        x = (x-torch.mean(x)) / (torch.std(x) + 0.0001)
    return x



def std_normalization(x):
    day_data = x[:,:,:5]
    day_data = (day_data - torch.mean(day_data))/(torch.std(day_data))
    week_data = x[:,:,5:10]
    week_data = (week_data - torch.mean(week_data))/(torch.std(week_data))
    x[:,:,:5] = day_data
    x[:,:,5:10] = week_data
    return x



def predict_normalization(x):

    _min = np.min(x)
    _range = np.max(x) - np.min(x)
    return (x - _min) / _range, _min, _range



def MinMax_normalization(x):
    if type(x) == torch.Tensor:
        ohlc, min_ohlc, range_ohlc = torch_scaler(x[:, 0:4])
        volume, min_v, range_v = torch_scaler(x[:, 4].reshape(-1, 1))
        x = torch.concat((ohlc,volume ), dim=1)
    else:
        ohlc, min_ohlc, range_ohlc = scaler(x[:, 0:4])
        volume , min_v, range_v = scaler(x[:, 4].reshape(-1, 1))
        x = np.concatenate((ohlc, volume), axis=1)
    return x, (min_ohlc, range_ohlc, min_v, range_v)

def MinMax_normalization_(x):
    length, channel = x.shape
    if channel==5:
        data, value = MinMax_normalization(x)
        return data, value,1
    if channel==10:
        day_data, day_value = MinMax_normalization(x[:, 0:5])
        week_data, week_value = MinMax_normalization(x[:, 5:10])
        if type(x) == torch.Tensor:
            data = torch.concat((day_data, week_data), dim=1)
        else:
            data = np.concatenate((day_data, week_data), axis=1)
        return data, day_value, week_value
    if channel==8:
        day = x[ :, :4]
        week = x[ :, 4:8]
        day,_,_ = torch_scaler(day)
        week,_,_ = torch_scaler(week)
        x[ :, :4] = day
        x[ :, 4:8] = week
        return x, None, None

def standard_normalization(x):
    if type(x) == torch.Tensor:
        ohlc = standard_torch_scaler(x[:, 0:4])
        volume= standard_torch_scaler(x[:, 4].reshape(-1, 1))
        x = torch.concat((ohlc,volume ), dim=1)
    else:
        ohlc = standard_scaler(x[:, 0:4])
        volume= standard_scaler(x[:, 4].reshape(-1, 1))
        x = np.concatenate((ohlc, volume), axis=1)
    return x

def standard_normalization_(x):
    length, channel = x.shape
    if channel==5:
        data = standard_normalization(x)
        return data
    if channel==10:
        day_data = standard_normalization(x[:, 0:5])
        week_data= standard_normalization(x[:, 5:10])
        if type(x) == torch.Tensor:
            data = torch.concat((day_data, week_data), dim=1)
        else:
            data = np.concatenate((day_data, week_data), axis=1)
        return data

def MinMax_denormalization_(x, day_value, week_value):
    batch,length, channel = x.shape
    d_min_ohlc, d_range_ohlc, d_min_v, d_range_v = day_value.squeeze()
    w_min_ohlc, w_range_ohlc, w_min_v, w_range_v = week_value.squeeze()
    if channel==5:
        ohlc = x[:,:, 0:4]
        volume = x[:,:, 4].unsqueeze(2)
        ohlc = torch.mul(ohlc, d_range_ohlc) + d_min_ohlc
        volume = torch.mul(volume, d_range_v) + d_min_v
        x = torch.concat((ohlc,volume ), dim=2)
        return x

    elif channel==10:
        d_ohlc = x[:,:, 0:4]
        d_volume = x[:,:, 4].unsqueeze(2)
        w_ohlc = x[:,:,5:9]
        w_volume = x[:,:, 9].unsqueeze(2)
        d_ohlc = torch.mul(d_ohlc, d_range_ohlc) + d_min_ohlc
        d_volume = torch.mul(d_volume, d_range_v) + d_min_v
        day = torch.concat((d_ohlc,d_volume ), dim=2)
        w_ohlc = torch.mul(w_ohlc, w_range_ohlc) + w_min_ohlc
        w_volume = torch.mul(w_volume, w_range_v) + w_min_v
        week = torch.concat((w_ohlc,w_volume ), dim=2)
        x = torch.concat((day,week ), dim=2)
        return x

def scaler(x):
    _min = np.min(x)
    _range = np.max(x) - np.min(x)
    return (x - _min) / _range, _min, _range

def standard_scaler(x):
    _mean = np.mean(x)
    _std = np.std(x)
    return (x - _mean) / _std

def standard_torch_scaler(x):
    _mean = torch.mean(x)
    _std = torch.std(x)
    return (x - _mean) / _std

def torch_scaler(x):
    _min = torch.min(x)
    _range = torch.max(x) - torch.min(x)
    return (x - _min) / _range, _min, _range



def denormalization(x, day_mean, day_std,day_mean_v, day_std_v, week_mean, week_std,week_mean_v,week_std_v):
    day_volume = x[:, :, 4]
    day_volume = day_volume * day_std_v + day_mean_v
    week_volume = x[:, :, 9]
    week_volume = week_volume * week_std_v + week_mean_v
    x[:, :, 4] = day_volume
    x[:, :, 9] = week_volume
    day_data = x[:, :, :4]
    day_data = day_data * day_std + day_mean
    week_data = x[:, :, 5:9]
    week_data = week_data * week_std + week_mean
    x[:, :, :4] = day_data
    x[:, :, 5:9] = week_data
    return x

def day_denormalization(x,day_mean, day_std):
    x = x * day_std + day_mean
    return x
def confirm_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

