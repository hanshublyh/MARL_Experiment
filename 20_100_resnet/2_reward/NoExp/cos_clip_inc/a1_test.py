from tqdm import tqdm
from modules import ActorCritic, CNNandDense, rl_utils
from src import Envn
import time
import psutil
import os
import matplotlib.pyplot as plt
import copy
import collections
import random
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import json
import math


ll_path = "../../"
rank_name = ll_path + 'rank.txt'

rank_array = np.loadtxt(rank_name, dtype=np.float32, delimiter=',')
rank = int(rank_array[2])

reward_kind_name = 'reward.txt'
with open(reward_kind_name, 'r') as f:
    reward_kind = f.read()

filename = ll_path + 'data' + str(rank) + '/'
dot = '.npy'
mac_kind = int(rank_array[0])
sev_kind = int(rank_array[1])

mac_sev_name = 'mac_sev' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
mac_trf_name = 'mac_trf' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
sev_trf_name = 'sev_trf' + '_' + str(sev_kind) + '_' + str(rank)  # 后面可以补充字符串说明
sev_trf_out_name = 'sev_trf_out' + '_' + \
    str(sev_kind) + '_' + str(rank)  # 后面可以补充字符串说明
sev_trf_in_name = 'sev_trf_in' + '_' + \
    str(sev_kind) + '_' + str(rank)  # 后面可以补充字符串说明
sev_cpu_name = 'sev_cpu' + '_' + str(sev_kind) + '_' + str(rank)
sev_mem_name = 'sev_mem' + '_' + str(sev_kind) + '_' + str(rank)
mac_cpu_name = 'mac_cpu' + '_' + str(mac_kind) + '_' + str(rank)
mac_mem_name = 'mac_mem' + '_' + str(mac_kind) + '_' + str(rank)
total_cpu_name = 'total_cpu' + '_' + str(mac_kind) + '_' + str(rank)
total_mem_name = 'total_mem' + '_' + str(mac_kind) + '_' + str(rank)
mac_sev_cpu_name = 'mac_sev_cpu' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
mac_sev_mem_name = 'mac_sev_mem' + '_' + \
    str(mac_kind) + '_' + str(sev_kind) + '_' + str(rank)
now_all_name = 'now_all' + '_' + str(mac_kind) + '_' + str(rank)

trf_cpu_mem = np.load(filename + now_all_name + dot)
mac_sev_cpu = np.load(filename + mac_sev_cpu_name + dot)
mac_sev_mem = np.load(filename + mac_sev_mem_name + dot)
mac_sev = np.load(filename + mac_sev_name + dot)
sev_trf = np.load(filename + sev_trf_name + dot)
sev_trf_out = np.load(filename + sev_trf_out_name + dot)
sev_trf_in = np.load(filename + sev_trf_in_name + dot)

s_cpu = np.load(filename + sev_cpu_name + dot)
s_mem = np.load(filename + sev_mem_name + dot)
stop = np.zeros(mac_sev.shape[0]).reshape((mac_sev.shape[0], 1))
trf_cpu_mem = np.concatenate((trf_cpu_mem, stop), axis=1)

y1 = mac_sev.sum(axis=0) * sev_trf_in
y2 = sev_trf_out * mac_sev.sum(axis=0).reshape((mac_sev.shape[1], 1))
print(np.allclose(y1, y2))
print(np.allclose(y1, sev_trf))

topk = 0  # np.loadtxt('line2.csv', delimiter=",")


print(trf_cpu_mem.shape)
print(mac_sev_cpu.shape)
print(mac_sev_mem.shape)
print(mac_sev.shape)
print(sev_trf.shape)
print(sev_trf_out.shape)
print(sev_trf_in.shape)
print(s_cpu.shape)
print(s_mem.shape)

# 参数列表 这里也应该读取文件
para_list = ['actor_lr : ', 'critic_lr : ', 'num_episodes : ', 'gamma : ', 'lmbda : ',
             'epochs : ', 'eps : ', 'deep : ', 'lenth : ', 'stop_r : ', 'sa : ', 'sb : ',
             'epoch_len : ', 'pround : ', 'step_r : ', 'traffic_r : ', 'bs : ']
para_path = '../../../'
with open(para_path + 'para1.txt', 'r', encoding='utf-8') as f:
    para = f.readlines()
    # print(para)
for i, p in enumerate(para):
    para[i] = eval(p.strip(para_list[i] + ',\n'))
    print((para_list[i], para[i]))

actor_lr = para[0]
critic_lr = para[1]
num_episodes = int(para[2])
gamma = para[3]
lmbda = para[4]
epochs = int(para[5])
eps = para[6]
deep = int(para[7])
lenth = int(para[8])
stop_r = para[9]
# reshape
s_l = int(para[10])
s_h = int(para[11])

step_r = para[14]
traffic_r = para[15]
bs = para[16]

with open(para_path + '../' + 'device.txt', 'r', encoding='utf-8') as f:
    device = torch.device(f.readlines()[0])
env_name = "VM——Reschedule"
env = Envn.Env(
    mac_sev, mac_sev_cpu, mac_sev_mem, trf_cpu_mem, sev_trf, sev_trf_out,
    sev_trf_in, reward_kind, s_cpu, s_mem, deep, lenth, topk, s_l, s_h, traffic_r)
# 随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

model = 'Actor-Critic'
act_dim = mac_sev.shape[0]

agent = ActorCritic.ActorCritic(act_dim, actor_lr, critic_lr,
                                gamma, lmbda, epochs, eps, device, model, num_episodes, deep, lenth, bs)


epoch_len = int(para[12])
per_len = int(num_episodes / epoch_len)
round = int(sum(sum(mac_sev)) / para[13])

print('num_episodes, actor_lr, critic_lr, gamma, lmbda, epochs, deep, lenth, round, reward, epoch_len, stop_r, eps, sl, sh step_r, traffic_r bs')
print(num_episodes, actor_lr, critic_lr, gamma, lmbda,
      epochs, deep, lenth, (para[13], round), reward_kind,
      epoch_len, stop_r, eps, s_l, s_h, step_r, traffic_r, bs)


b_stop = 0
b_stop_trf = 0
b_trf = 0
b_trf_stop = 0
cnt = 0

punish_list = np.zeros(num_episodes)
reward_list = np.zeros(num_episodes)
dids_list = np.zeros(num_episodes)
return_list = np.zeros(num_episodes)
trf_list = np.zeros((num_episodes, env.mac_sev.shape[0]))
cpus = np.zeros((num_episodes, env.mac_sev.shape[0]))
mems = np.zeros((num_episodes, env.mac_sev.shape[0]))
stop_list = np.zeros((num_episodes, env.mac_sev.shape[0]))
mac_sev_list = np.zeros((num_episodes, mac_sev.shape[0], mac_sev.shape[1]))
for i in range(epoch_len):  # 分几个进度条显示
    with tqdm(total=per_len, desc='Iteration %d' % i) as pbar:
        for i_episode in range(per_len):

            # start = time.perf_counter()
            transition_dict = {
                'q3': [],
                'act_mac2s': [],
                'act_mac1s': [],
                'q3_': [],
                'rewards': [],
                'dids': [],
                'dones': [],
                # 'mac2': [],
                'mac_sev': [],
                'pi_3': [],

            }
            state = env.reset()
            done = False

            while not done:
                env.the_now_mac_sev_pos_1agent()
                transition_dict['act_mac1s'].append(env.m)
                mask = env.mac_2_mask()

                action, q_value, pi = agent.take_action(
                    state, mask, True, env, 3)

                next_state, reward, did, done = env.step_1a(action)
                tmp = torch.tensor(
                    next_state, dtype=torch.float).to(device).clone()
                tmp = torch.unsqueeze(tmp, 0).to(device)
                q_value_ = agent.critic(tmp)

                transition_dict['q3'].append(q_value)
                transition_dict['act_mac2s'].append(int(action))
                transition_dict['dids'].append(int(did))
                transition_dict['q3_'].append(q_value_)
                transition_dict['rewards'].append(reward.item())
                transition_dict['dones'].append(done)
                transition_dict['pi_3'].append(list(pi[0]))
                # print(pi[0].shape)
                # transition_dict['mac2'].append(mask)

                # 如果在这里才存储 那么就是没初始状态 因为是更新后的 不过最后一个肯定是最终态
                transition_dict['mac_sev'].append(env.mac_sev)

                state = next_state

            # 输出对不对
            # rl_utils.TEST_INSTANCE(env, mac_sev, s_cpu, s_mem, transition_dict)
            #########################################
            # print(mac_sev.sum())
            # 测试本次调度对不对 主要是用mac_sev的值和s_cpu s_mem
            print('步数对不对 dids states next')
            print(len(transition_dict['dids']) == sum(sum(mac_sev)))
            print(len(transition_dict['q3']) == sum(sum(mac_sev)))
            print(len(transition_dict['q3_']) == sum(sum(mac_sev)))
            print('符不符合实际 mac_sev没有为负数的 且服务种类还是一样')
            print(np.allclose(env.mac_sev.sum(axis=0), mac_sev.sum(axis=0)))

            its_a_test = 0
            for a in range(env.mac_sev.shape[0]):
                for b in range(env.mac_sev.shape[1]):
                    if env.mac_sev[a][b] < 0:
                        its_a_test = 1
                        print('有错误')
            if (its_a_test == 0):
                print('没错误')

            print('TEST: CPU MEM TRF')
            # cpu对不对
            mac1 = np.zeros((mac_sev.shape[0]))
            for a in range(mac_sev.shape[0]):
                for b in range(mac_sev.shape[1]):
                    mac1[a] += env.mac_sev[a][b] * s_cpu[b]
            print(np.allclose(mac1, env.cpus))
            # mem对不对
            mac2 = np.zeros((mac_sev.shape[0]))
            for a in range(mac_sev.shape[0]):
                for b in range(mac_sev.shape[1]):
                    mac2[a] += env.mac_sev[a][b] * s_mem[b]
            print(np.allclose(mac2, env.mems))
            # trf对不对
            mact = np.zeros((mac_sev.shape[0]))
            for a in range(env.mac_sev.shape[0]):
                for b in range(env.mac_sev.shape[1] - 1):
                    for c in range(b + 1, env.mac_sev.shape[1]):
                        mact[a] += min(env.sev_trf_out[b][c] * env.mac_sev[a]
                                       [b], env.sev_trf_in[b][c] * env.mac_sev[a][c])
            print(np.allclose(mact, env.now_trf))
            ############################################
            # 加入stop情况
            stop_list[cnt] = env.stop
            # 加入cpu
            cpus[cnt] = env.cpus
            # 加入mems
            mems[cnt] = env.mems
            # 就是一次只存now——trf 然后读到文件里 最后再从文件一个一个读出来 可以不用存整个矩阵了
            trf_list[cnt] = env.now_trf

            if sum(env.stop) > b_stop or (sum(env.stop) == b_stop and sum(env.now_trf) > b_stop_trf):
                torch.save(agent.actor.state_dict(),
                           './model/a1a_bt_under_bs.pt')
                torch.save(agent.critic.state_dict(),
                           './model/a1c_bt_under_bs.pt')
                b_stop = sum(env.stop)
                b_stop_trf = sum(env.now_trf)

                for tmp in range(0, len(transition_dict['mac_sev']), 15):
                    np.savetxt('./BS_state_change_a1/BS_' + str(tmp) + '.csv',
                               transition_dict['mac_sev'][tmp], delimiter=',')
                np.savetxt('./BS_state_change_a1/BS_last' + '.csv',
                           transition_dict['mac_sev'][-1], delimiter=',')
                
            if sum(env.now_trf) > b_trf or (sum(env.stop) > b_stop and sum(env.now_trf) == b_stop_trf):
                torch.save(agent.actor.state_dict(),
                           './model/a1a_bs_under_bt.pt')
                torch.save(agent.critic.state_dict(),
                           './model/a1c_bs_under_bt.pt')
                b_trf = sum(env.now_trf)
                b_trf_stop = sum(env.stop)

            com_reward, punish = agent.update(transition_dict, 'where', sum(env.stop), stop_r,
                                              reward_kind, step_r, stop_list[cnt], list(env.hcpus), list(env.hmems))
            print(sum(env.now_trf) / sum(env.ini_trf))
            return_list[cnt] = sum(com_reward)
            mac_sev_list[cnt] = env.mac_sev
            punish_list[cnt] = punish
            np.savetxt('./last_state_a1/' + str(cnt) + 'last' +
                       '.csv', env.mac_sev, delimiter=',')
            del transition_dict
            if (i_episode + 1) % per_len == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (per_len * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[cnt + 1 - per_len: cnt + 1])
                })
            pbar.update(1)
            cnt += 1
            # 那样可以多次移动
            '''
            end = time.perf_counter()
            # 计算运行时间
            runTime = end - start
            f = open('runtime1.txt', 'a')
            f.writelines('\n' + str((i * per_len + i_episode + 1, runTime)) + '\n')
            f.close()'''

print(dids_list)
print(punish_list)
rl_utils.show(env, stop_list, trf_list, dids_list, mac_kind, sev_kind,
              mac_sev_list, cpus, mems, return_list, rank, reward_kind, 1, para[13], 'para1', punish_list)

del stop_list
del trf_list
del dids_list
del punish_list
del mac_sev_list
del return_list
del cpus
del mems
