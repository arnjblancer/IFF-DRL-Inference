import argparse
import os
from datetime import datetime

import pandas as pd
import yfinance as yf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataLoad.DataLoaderRL import RLDataSet, stock_fn
from dataLoad.utils import normalization
from agents.DQNagent import DDQN
from Env.TradingEnv import TradingEnv
from utils.estimator import PerformanceEstimator
from utils.utils import setup_seed


def download_with_yfinance(ticker: str, start: str, end: str, path: str) -> None:
    """Download historical OHLCV data and store as CSV."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)


def train_and_backtest(args) -> None:
    setup_seed(1)
    cfg = OmegaConf.load("configs/common.yaml")
    cfg.dataSetName = args.ticker
    cfg.ValDataName = args.ticker
    cfg.time.train_startingDate = args.train_start
    cfg.time.train_endingDate = args.train_end
    cfg.time.test_startingDate = args.test_start
    cfg.time.test_endingDate = args.test_end
    cfg.agent.ModelPath = os.path.join("result", f"{args.ticker}_{cfg.agent.algorithm}.pth")
    cfg.device = "cpu"

    data_path = os.path.join(cfg.dataDir, f"{args.ticker}.csv")
    if not os.path.exists(data_path):
        download_with_yfinance(args.ticker, args.train_start, args.test_end, data_path)

    train_dataset = RLDataSet(
        csvFilePath=data_path,
        startDate=datetime.strptime(args.train_start, "%Y-%m-%d"),
        endDate=datetime.strptime(args.train_end, "%Y-%m-%d"),
        cfg=cfg,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=stock_fn)

    test_dataset = RLDataSet(
        csvFilePath=data_path,
        startDate=datetime.strptime(args.test_start, "%Y-%m-%d"),
        endDate=datetime.strptime(args.test_end, "%Y-%m-%d"),
        cfg=cfg,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=stock_fn)

    env = TradingEnv(cfg, init_money=cfg.train.init_money)
    agent = DDQN(cfg)

    for epoch in range(args.epochs):
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
    parser = argparse.ArgumentParser(description="Train and backtest an RL trading agent")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--train_start", required=True, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train_end", required=True, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--test_start", required=True, help="Testing start date (YYYY-MM-DD)")
    parser.add_argument("--test_end", required=True, help="Testing end date (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()
    train_and_backtest(args)
