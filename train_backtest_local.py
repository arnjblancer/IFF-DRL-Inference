import os
from datetime import datetime

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataLoad.DataLoaderRL import RLDataSet, stock_fn
from dataLoad.utils import normalization
from agents.DQNagent import DDQN
from Env.TradingEnv import TradingEnv
from utils.estimator import PerformanceEstimator
from utils.utils import setup_seed


def train_and_backtest() -> None:
    """Train an agent on a local CSV file and run a backtest.

    All configuration is loaded from ``configs/common.yaml`` without
    additional command-line overrides.
    """
    setup_seed(1)
    cfg = OmegaConf.load("configs/common.yaml")

    data_path = os.path.join(cfg.dataDir, f"{cfg.dataSetName}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found")

    train_dataset = RLDataSet(
        csvFilePath=data_path,
        startDate=datetime.strptime(cfg.time.train_startingDate, "%Y-%m-%d"),
        endDate=datetime.strptime(cfg.time.train_endingDate, "%Y-%m-%d"),
        cfg=cfg,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=stock_fn)

    test_dataset = RLDataSet(
        csvFilePath=data_path,
        startDate=datetime.strptime(cfg.time.test_startingDate, "%Y-%m-%d"),
        endDate=datetime.strptime(cfg.time.test_endingDate, "%Y-%m-%d"),
        cfg=cfg,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=stock_fn)

    env = TradingEnv(cfg, init_money=cfg.train.init_money)
    agent = DDQN(cfg)

    os.makedirs(os.path.dirname(cfg.agent.ModelPath), exist_ok=True)

    for epoch in range(cfg.train.epochs):
        env.reset(len(train_loader))
        for iteration, batch in enumerate(train_loader):
            datas, price_tm1, price_t, price_tpn, datas_next, mask, mask_next = [i.to(cfg.device) for i in batch]
            if cfg.normalization:
                datas = normalization(datas)
                datas_next = normalization(datas_next)
            if iteration == 0:
                env.setInitPrice(price_t)
            env.setState(iteration, price_tm1, price_t, price_tpn)
            env.getNday()
            action = agent.act(datas, train=True)
            reward = env.agentStep(datas, action)
            agent.memory.Agent_add(datas, action, reward, datas_next)
            if len(agent.memory) > cfg.train.batch_size and iteration % cfg.train.train_interval == 0:
                batch_sample = agent.memory.Agent_sample(cfg.train.batch_size)
                agent.learn(batch_sample)
        torch.save(agent.policy_net.state_dict(), cfg.agent.ModelPath)

    agent.policy_net.eval()
    env.reset(len(test_loader))
    for iteration, batch in enumerate(test_loader):
        datas, price_tm1, price_t, price_tpn, datas_next, mask, mask_next = [i.to(cfg.device) for i in batch]
        if cfg.normalization:
            datas = normalization(datas)
        if iteration == 0:
            env.setInitPrice(price_t)
        env.setState(iteration, price_tm1, price_t, price_tpn)
        env.getNday()
        action = agent.act(datas, train=False)
        env.agentStep(datas, action)

    analyser = PerformanceEstimator(cfg, env.account, "result")
    annual_ret = analyser.computeAnnualizedReturn(len(test_loader))
    total_ret = analyser.computeCR() * 100
    max_dd, _ = analyser.computeMaxDrawdown()
    print(f"Annualized Return: {annual_ret:.2f}%")
    print(f"Total Return: {total_ret:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}%")


if __name__ == "__main__":
    train_and_backtest()
