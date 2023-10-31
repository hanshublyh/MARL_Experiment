import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.categorical as Categorical
import numpy as np
import pandas as pd
import random
import copy
import time
from modules import rl_utils
import modules.CNNandDense as Net


class PPO:
    def __init__(self, act_dim, actor_lr, critic_lr, gamma, lamda,
                 K_epochs, eps, device, max_steps, depth, lenth, batch_size) -> None:

        # action_space
        self.act_dim = act_dim

        # CPU or GPU CUDA
        self.device = device

        # ppo parameter
        self.gamma = gamma  # gamma
        self.lamda = lamda  # lamda
        self.eps = eps  # clip_rate
        self.K_epochs = K_epochs  # update times

        self.steps = max_steps  # max_steps to update the lr

        self.Batch_Size = batch_size  # batch_size for per round training

        # actor Net
        self.actor = Net.ActCNN(act_dim, depth, lenth).to(self.device)
        self.actor_lr = actor_lr
        self.actor_opt = optim.Adam(lr=self.actor_lr)
        # critic Net
        self.critic = Net.ValueCNN(1, depth, lenth).to(self.device)
        self.critic_lr = critic_lr
        self.critic_opt = optim.Adam(lr=self.critic_lr)

        

    def take_actions(self, state, mask, Use_mask, env):
        # state : np.array
        # single input shape : eg. (3, 224, 224)
        # reshape -- > (1, 3, 224, 224)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = torch.unsqueeze(state, 0)

        if Use_mask:
            # if all the acts have been masked but there is no such an occasion
            # then the agent will be forced to select the current mac
            if mask.sum() == 0:
                act = env.m
            else:
                mask = torch.tensor(mask).to(self.device)
                pure_prob = self.actor(state)
                mask_prob = pure_prob * mask

                distribution = Categorical(mask_prob)
                act = distribution.sample().item()
                pi_a = distribution[act]

        return act, pi_a
    
    def train(self):

    # call in main.py
    def collect_data(self, transition, agent):
        

        

    # call self.collect_batch() in self.train()
    def collect_batch(self):
        l = len(self.data)
        for i in range()

