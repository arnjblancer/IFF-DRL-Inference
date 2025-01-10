import os
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataLoad.DataLoaderRL import RLDataSet, stock_fn

from utils.utils import setup_seed
from trainers.Tester import AgentTester
from dataLoad.utils import get_date
from Model.TimesNet import TimesNet
import pandas as pd
setup_seed(1)
cfg = OmegaConf.load('configs/common.yaml')
train_startingDate = datetime.strptime(cfg.time.train_startingDate, '%Y-%m-%d')
train_endingDate = datetime.strptime(cfg.time.train_endingDate, '%Y-%m-%d')
test_startingDate = datetime.strptime(cfg.time.test_startingDate, '%Y-%m-%d')
test_endingDate = datetime.strptime(cfg.time.test_endingDate, '%Y-%m-%d')


filesPaths = os.path.join(cfg.dataDir, cfg.dataSetName + '.csv')
valFilePath = os.path.join(cfg.dataDir, cfg.ValDataName + '.csv')


RLTrainDataSet = RLDataSet(csvFilePath=filesPaths,
                                      startDate=train_startingDate,
                                      endDate=train_endingDate,
                                      cfg=cfg
                                      )
RL_train_data = DataLoader(RLTrainDataSet, batch_size=1,
                        collate_fn=stock_fn
                        )


RLValDataSet = RLDataSet(csvFilePath=valFilePath,
                                    startDate=test_startingDate,
                                    endDate=test_endingDate,
                                    cfg=cfg
                                    )
RL_val_data = DataLoader(RLValDataSet, batch_size=1,
                      collate_fn=stock_fn
                      )


Tester = AgentTester(cfg, AgentName='agent')
Tester.test(RL_val_data, isTrain=False)





