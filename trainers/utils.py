import torch
from utils.augmentations import masked_data
import torch.nn as nn
import random
import pandas as pd
def getBatch(batch):
    assert len(batch) == 5 or 3
    if len(batch) == 6:
        index, state, trend, mask, minValue,rangeValue = batch
        return index, state, trend, mask, minValue,rangeValue
    if len(batch) == 3:
        state, trend, mask = batch
        return state, trend, mask
def analyze_model_structure(model):
    module_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in module_counts:
            module_counts[module_type] += 1
        else:
            module_counts[module_type] = 1
    return module_counts
def getMask(batch_x,batch_x_mark,cfg, netCfg):
    if netCfg.select_channels < 1:

        # random select channels
        B, S, C = batch_x.shape
        random_c = int(C * netCfg.select_channels)
        if random_c < 1:
            random_c = 1

        index = torch.LongTensor(random.sample(range(C), random_c))
        batch_x = torch.index_select(batch_x, 2, index)
    batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, netCfg.mask_rate,
                                                  netCfg.lm, netCfg.positive_nums)
    batch_x_om = torch.cat([batch_x, batch_x_m], 0)

    batch_x = batch_x.float().to(cfg.device)
    # batch_x_mark = batch_x_mark.float().to(self.device)

    # masking matrix
    mask = mask.float().to(cfg.device)
    mask_o = torch.ones(size=batch_x.shape).float().to(cfg.device)
    mask_om = torch.cat([mask_o, mask], 0).to(cfg.device)

    # to device
    batch_x = batch_x.float().to(cfg.device)
    batch_x_om = batch_x_om.float().to(cfg.device)
    batch_x_mark = batch_x_mark.float().to(cfg.device)

    return batch_x_om, batch_x_mark, batch_x, mask_om


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    xb = xb[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1,
                                                                        D))  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                            device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]
    a = ids_restore.unsqueeze(-1).repeat(1, 1, 1,D)
    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1,
                                                                              D))  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask



def tail_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_mask = int(L * mask_ratio)
    len_keep = L - len_mask

    # 保留前 len_keep 部分
    x_kept = x[:, :len_keep, :, :]  # 保留部分的张量，形状为 [bs x len_keep x nvars x patch_len]

    # 掩码后面 len_mask 部分
    x_removed = torch.zeros(bs, len_mask, nvars, D, device=xb.device)  # 掩码部分的全零张量，形状为 [bs x len_mask x nvars x patch_len]

    # 组合保留部分和掩码部分
    x_masked = torch.cat([x_kept, x_removed], dim=1)  # 拼接保留部分和掩码部分，形状为 [bs x L x nvars x patch_len]

    # 生成二进制掩码：0 是保留，1 是移除
    mask = torch.ones([bs, L, nvars], device=x.device)  # 创建一个全一张量表示掩码
    mask[:, :len_keep, :] = 0  # 将保留部分设置为0

    return x_masked, x_kept, mask

class MSEWithL2Regularization(nn.Module):
    def __init__(self, weight_decay=0.01):
        super(MSEWithL2Regularization, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.weight_decay = weight_decay

    def forward(self, predictions, targets, parameters):
        mse_loss = self.mse_loss(predictions, targets)
        l2_reg = 0.0
        for param in parameters:
            l2_reg += torch.norm(param) ** 2  # 计算L2正则化项
        return mse_loss + self.weight_decay * l2_reg

def generate_dayweek(data):
    data['Date'] = pd.to_datetime(data['Date'])
    day_data = data.dropna()

    day_data = day_data[-data.Volume.isin([0])]

    # day_data['Date'] = pd.to_datetime(day_data['Date'])
    # day_data = day_data.loc[(day_data['Date'] >= start_date) & (day_data['Date'] <= end_date)]

    week = day_data  # 获取和日数据一样形状的dataframe
    #day_data = day_data.drop(['Close'], axis=1)
    # day_data = day_data.drop(['Date'], axis=1)
    day_data.reset_index(drop=True, inplace=True)

    week_group = data.dropna()
    week_group = week_group[-data.Volume.isin([0])]
    # week_group = week_group.loc[(week_group['Date'] >= start_date) & (week_group['Date'] <= end_date)]
    # week_group['Date'] = pd.to_datetime(week_group['Date'])
    week_group = week_group.resample('W-FRI', on='Date')
    week_data = week_group.last()

    week_data['Open'] = week_group.first()['Open']
    week_data['Low'] = week_group.min()['Low']
    week_data['High'] = week_group.max()['High']
    week_data['Volume'] = week_group.sum()['Volume']
    week_data['Date'] = week_group['Date']
    week_data['Date'] = week_data.index
    week_data.rename(columns={'Date': 'temp'}, inplace=True)
    week_data.insert(0, 'Date', week_data['temp'])
    week_data.drop(axis=1, columns=['temp'], inplace=True)
    j = 0
    for i in range(len(week)):  # 对比日数据和周数据的日期，将日期少于周数据的替换成周数据
        if week.iloc[i]['Date'] <= week_data.iloc[j]['Date']:
            week.iloc[i] = week_data.iloc[j]
        else:
            j += 1
            week.iloc[i] = week_data.iloc[j]
    week.reset_index(drop=True, inplace=True)

    if 'Original Price' not in week.columns:
        week['Original Price'] = week['Adj Close']
    week = week[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Original Price']]
    #day_data.drop('Original Price', axis=1, inplace=True)
    input_data = pd.concat([data, week], axis=1).reset_index(drop=True)
    input_data['Volume'] = input_data['Volume'].astype(int)
    return input_data
def concat_DayWeek(cfg, data,predict,name, start_point):
    """
    data: 日数据表
    predict: 预测的OHLCV
    """
    data['Date'] = pd.to_datetime(data['Date'])
    if cfg.train.learn:
        original_dayweek = pd.read_csv('PredictData/{}.csv'.format(name))
        dayweek = original_dayweek.loc[(original_dayweek['Date'] >= cfg.time.test_startingDate)]
        data = data.loc[(data['Date'] >= cfg.time.test_startingDate) & (data['Date'] <= cfg.time.test_endingDate)]
    else:
        original_dayweek = pd.read_csv('DayWeekData/{}.csv'.format(name)) #加载日周数据
        dayweek = original_dayweek
        data = data.loc[(data['Date'] >= cfg.time.train_startingDate) & (data['Date'] <= cfg.time.test_endingDate)]



    data = data.dropna()

    data = data[-data.Volume.isin([0])]

    #day_data['Date'] = pd.to_datetime(day_data['Date'])
    #day_data = day_data.loc[(day_data['Date'] >= start_date) & (day_data['Date'] <= end_date)]
    if 'Close'in data.columns:
        data = data.drop(['Close'], axis=1)

    #day_data = day_data.drop(['Date'], axis=1)
    data.reset_index(drop=True, inplace=True)

    j = 0
    for i in range(len(data)-4):#对比日数据和周数据的日期，将日期少于周数据的替换成周数据
        if i<start_point:
            continue
        data.iloc[i:i+5,1:6] = predict[j]
        j+=1
        '''
                if data['Date'][i].isoweekday()<=5:
            data.iloc[i,6:10] = predict[j]
            if data['Date'][i+1].isoweekday()<=data['Date'][i].isoweekday():
                j+=1
        '''


    data.reset_index(drop=True, inplace=True)
    data = generate_dayweek(data)
    dayweek.iloc[:,6:11] = data.iloc[:,6:11]

    if cfg.train.learn:
        original_dayweek.loc[(original_dayweek['Date'] >= cfg.time.test_startingDate) & (
                original_dayweek['Date'] <= cfg.time.test_endingDate)] = dayweek
        original_dayweek.to_csv('Incremental_Data_20-23/{}_test.csv'.format(name), index=False)
    else:
        original_dayweek.loc[(original_dayweek['Date'] >= cfg.time.train_startingDate) & (
                original_dayweek['Date'] <= cfg.time.test_endingDate)] = dayweek
        original_dayweek.to_csv('PredictData/{}_test.csv'.format(name), index=False)