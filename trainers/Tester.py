import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.estimator import PerformanceEstimator
from utils.ReplayBuffer import ReplayBuffer
from agents.ACagent import A2C
from agents.PPOagent import PPO
from agents.DQNagent import DDQN
from Env.TradingEnv import TradingEnv
from dataLoad.utils import normalization
from utils.RewardRecord import RewardRecord
from dataLoad.utils import confirm_makedirs
import os
from utils.Visualizer import Visualizer

class AgentTester:

    def __init__(self, cfg, AgentName):
        """
        :param cfg: common.yaml
        :param AgentName: 模型名称，用来保存训练文件和加载配置文件（.yaml）
        """
        self.cfg = cfg
        self.env = TradingEnv(cfg, init_money=cfg.train.init_money)
        self.agent = eval(cfg.agent.algorithm)(cfg)
        self.memory = ReplayBuffer(self.cfg.agent.buffer_size)
        self.saveDir = os.path.join('result', cfg.expName, AgentName)
        self.agentName = AgentName
        self.saveModelDir = os.path.join(self.saveDir, 'model')
        self.testDir = os.path.join(self.saveDir, 'test')
        confirm_makedirs(self.saveModelDir)
        confirm_makedirs(self.testDir)
        self.visualizer = Visualizer()
        self.lossRecord = RewardRecord(self.testDir,self.cfg)
        self.agent.policy_net.load_state_dict(
            torch.load(self.cfg.agent.ModelPath, map_location=torch.device(cfg.device)))
        self.agent.policy_net.eval()
        self.best = -999999
        self.num_count= []
    def test(self, val_data, isTrain=False):
        self.agent.policy_net.eval()
        totalStep = len(val_data)
        self.env.reset(totalStep)
        TotalReward = 0
        with tqdm(total=totalStep, desc='TestStep') as pbar:
            for iteration, batch in enumerate(val_data):
                datas, priceTMinus1, priceT, priceTPlusN, datasNext, mask, maskNext = [i.to(self.cfg.device) for i in batch]
                if self.cfg.normalization:
                    datas = normalization(datas)
                if iteration == 0:
                    self.env.setInitPrice(priceT)
                self.env.setState(iteration, priceTMinus1, priceT, priceTPlusN)
                self.env.getNday()
                if self.cfg.agent.algorithm == 'PPO':
                    agent_action,_= self.agent.act(datas, train=False)
                else:
                    agent_action = self.agent.act(datas, train=False)
                reward = self.env.agentStep(datas, agent_action)
                TotalReward += reward.item()
                pbar.update(1)

        self.lossRecord.CapitalPlot(self.env.account, self.agentName, isTrain)
        self.lossRecord.draw(self.env.account, self.agentName)
        analyser = PerformanceEstimator(self.cfg, self.env.account, self.testDir)
        analyser.displayPerformance(self.agentName, len(val_data))
