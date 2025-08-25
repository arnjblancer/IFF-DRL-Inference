# Incremental Forecast Fusion Deep Reinforcement Learning

This project is a reinforcement learning-based stock trading system. It allows you to train and test models to optimize stock trading strategies.

## Dependencies

Before running the project, make sure you have the following dependencies installed:

* Python 3.9.12
* PyTorch 2.8.0
* OmegaConf
* Tqdm
* Numpy
* Pandas
* Matplotlib
* Tabulate
* Pillow

To execute the code, simply run the following command:

```bash
python main.py
```

## Train and backtest on local data

Use `train_backtest_local.py` to train a DDQN agent on a CSV file located in `Incremental_Data` and evaluate its performance.

Example:
```bash
python train_backtest_local.py --data_name N225 \
    --train_start 2020-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2021-12-31 \
    --epochs 1
```

The script prints the annualized return, total return, and maximum drawdown after the backtest.
