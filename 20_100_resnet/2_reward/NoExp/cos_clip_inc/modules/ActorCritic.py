import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import copy
import collections
import time
from modules import CNNandDense, rl_utils
import gc
import psutil
import os
import math


class ActorCritic:
    def __init__(self, act_dim, actor_lr, critic_lr, gamma, lmbda, epochs, eps, device, model, epo, deep, lent, bs):
        self.act_dim = act_dim
        self.deep = deep
        self.actor = CNNandDense.ActCNN(act_dim, deep, lent).to(device)
        self.critic = CNNandDense.ValueCNN(1, deep, lent).to(device)
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.cirloss = nn.MSELoss()
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        # self.total_opt = optim.Adam(self.ac.parameters(), lr = self.actor_lr)
        self.act_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.cri_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.device = device
        self.model = model
        self.now_epo = 0
        self.total_epo = epo
        # self.topk = topk
        self.bs = bs

    def update_lr(self):
        self.now_epo += 1

        for param_group in self.act_opt.param_groups:
            param_group['lr'] = self.actor_lr * \
                (1 - self.now_epo / self.total_epo)
        for param_group in self.cri_opt.param_groups:
            param_group['lr'] = self.critic_lr * \
                (1 - self.now_epo / self.total_epo)

    def take_action(self, state, mask, has_mask, env, agent):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = torch.unsqueeze(state, 0).to(self.device)

        if has_mask == True:
            # 选mac1 sev 不可能出现全mask 但是选mac2很可能全部mask了 所以如果全mask act == env.m
            # 但是自己不mask 所以永远会有自己 不会出现全部mask
            if sum(mask) == 0:
                act = env.m
            else:
                prob = self.actor(state) * torch.tensor(mask).to(self.device)
                # 确定性策略
                distribution = torch.distributions.Categorical(prob)
                act = distribution.sample().item()
        else:
            prob = self.actor(state)
        # 如果选了自己 后面还要在env判断一

        q_value = self.critic(state)
        # print(q_value.shape)
        # print(prob.shape)
        return act, q_value, self.actor(state)
# 为什么这样之后梯度就没了？和纯这样的对比一下每一个是不是一样

    def update(self, transitions, agent_name, stop, per_reward, kinds, step_r, stop_list, cpus, mems):
        if (agent_name == 'which'):
            actions_name = 'act_sevs'  # sev
            pi_name = 'pi_2'
            q_name = 'q2'
            q_name_ = 'q2_'
        elif (agent_name == 'mac'):
            actions_name = 'act_mac1s'  # mac1
            pi_name = 'pi_1'
            q_name = 'q1'
            q_name_ = 'q1_'
        elif (agent_name == 'where'):
            actions_name = 'act_mac2s'  # mac2
            pi_name = 'pi_3'
            q_name = 'q3'
            q_name_ = 'q3_'

        q_values = transitions[q_name]
        actions = transitions[actions_name]
        dids = transitions['dids']
        q_values_ = transitions[q_name_]
        pis = transitions[pi_name]
        dones = transitions['dones']
        rewards = transitions['rewards']
        the_actions1 = transitions['act_mac1s']
        the_actions3 = transitions['act_mac2s']

        punish = 0
        index_list = []
        cpus_list = []
        mems_list = []
        # print(stop_list)
        # print(cpus)
        # print(mems)
        for i in range(stop_list.shape[0]):
            if stop_list[i] == 1:
                index_list.append(i)
                cpus_list.append(cpus[i])
                mems_list.append(mems[i])
        # print(cpus_list)
        # print(mems_list)
        if cpus_list == [] or mems_list == []:
            reward_st = 0
        else:
            cpus_list = cpus_list / max(cpus_list)
            mems_list = mems_list / max(mems_list)
            p_list = (cpus_list + mems_list) / 2
            reward_st = sum(per_reward * p_list)
        del cpus_list
        del mems_list

        for i in range(stop_list.shape[0]):
            if stop_list[i] == 1:
                index_list.append(i)

        if kinds == 'r3':  # r3
            for i in range(len(rewards)):
                if dids[i] == 1 and (the_actions1[i] in index_list) and (the_actions3[i] not in index_list):
                    rewards[i] += reward_st

                if dids[i] == 1 and ((the_actions1[i] not in index_list) or (the_actions3[i] in index_list)):
                    rewards[i] -= step_r
                    punish += 1

        elif kinds == 'r1 + r2':  # r1 + r2
            for i in range(len(rewards)):
                if dids[i] == 1 and (the_actions1[i] in index_list) and (the_actions3[i] not in index_list):
                    rewards[i] += reward_st

                if dids[i] == 1 and ((the_actions1[i] not in index_list) or (the_actions3[i] in index_list)):
                    punish += 1

        elif kinds == 'r1':
            for i in range(len(rewards)):
                if dids[i] == 1 and (the_actions1[i] in index_list) and (the_actions3[i] not in index_list):
                    rewards[i] += reward_st

                if dids[i] == 1 and ((the_actions1[i] not in index_list) or (the_actions3[i] in index_list)):
                    punish += 1

        elif kinds == 'r2':
            for i in range(len(rewards)):
                if dids[i] == 1 and ((the_actions1[i] not in index_list) or (the_actions3[i] in index_list)):
                    punish += 1

        elif kinds == 'cos':
            for i in range(len(rewards)):
                if dids[i] == 1 and ((the_actions1[i] not in index_list) or (the_actions3[i] in index_list)):
                    punish += 1

        if self.model == 'Actor-Critic':
            # PPO
            # reward shaping
            # shuffle data not using every data to update? need a Buffer?
            # exploration and exploitation need after-judge?
            # Experience Replay???
            # mask feedback?
            # on-policy and off-policy?? needs a buffer?
            BATH_SIZE = self.bs
            lenth = len(q_values)
            rnds = math.ceil(lenth / BATH_SIZE)

            for rnd in range(rnds):
                rnd_q_values = torch.tensor(q_values[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)],
                                            dtype=torch.float).view(-1, 1).to(self.device)
                rnd_q_values.requires_grad_(True)
                rnd_q_values_ = torch.tensor(q_values_[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)],
                                             dtype=torch.float).view(-1, 1).to(self.device)
                # print(pis.shape)
                rnd_pis = torch.tensor(pis[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)],
                                       dtype=torch.float).to(self.device)
                rnd_pis.requires_grad_(True)
                rnd_act = torch.tensor(
                    actions[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)]).view(-1, 1).to(self.device)
                rnd_rewards = torch.tensor(rewards[rnd * BATH_SIZE: min(
                    (rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).view(-1, 1).to(self.device)
                rnd_dones = torch.tensor(
                    dones[rnd * BATH_SIZE: min((rnd + 1) * BATH_SIZE, lenth)], dtype=torch.float).view(-1, 1).to(self.device)

                rnd_q_target = rnd_rewards + self.gamma * \
                    rnd_q_values_ * (1 - rnd_dones)
                rnd_delta = rnd_q_target - rnd_q_values

                # 如果你给的概率很大 比如mac2 选择 5 概率很大， 但是5被mask了 所以最终的reward==0
                # 那么我的疑问是 这个prob需不需要认为修改？修改了之后回传有用吗？也就是 prob * did 但是自己是可以选择的
                # 还有一个问题就是 给了非法动作之后，也就是说最终不可能选出来非法动作 因为是0了嘛（非法就是资源不够）除了选择自己
                # 这个关键问题就是我想要你能够把非法动作预测的概率降低 只靠reward为0 而没有手动修改prob的值 回传行不行？
                probs = (rnd_pis).gather(1, rnd_act)
                # print(probs)
                # print(rnd_pis.shape)
                # print(probs.shape)
                log_probs = torch.log(probs + 1e-10)
                # log_probs.requires_grad_(True)

                # 目前看似成了off-policy 但是更新参数的时候用的还是一个策略
                actor_loss = torch.mean(-log_probs * rnd_delta.detach())
                critic_loss = torch.mean(
                    F.mse_loss(rnd_q_values, rnd_q_target.detach()))
                # actor_loss.requires_grad_(True)
                # critic_loss.requires_grad_(True)
                self.act_opt.zero_grad()
                self.cri_opt.zero_grad()

                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 5, norm_type=2)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 5, norm_type=2)
                self.act_opt.step()
                self.cri_opt.step()

                del rnd_q_values
                del rnd_q_values_
                del rnd_pis
                del rnd_act
                del rnd_delta
                del rnd_rewards
                del rnd_q_target

            self.update_lr()

        del q_values_
        del q_values
        del transitions
        del actions
        del dids
        del dones
        del the_actions1
        del the_actions3
        del pis
        return rewards, punish
