#coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections
import copy
import matplotlib.pyplot as plt
import os
import time
from src import Envn
from modules import ActorCritic, CNNandDense, rl_utils
from tqdm import tqdm

ll_path = "../../"
rank_name = ll_path + 'rank.txt'
VT_name = 1

rank_array = np.loadtxt(rank_name, dtype=np.float32, delimiter=',')
rank = int(rank_array[2])

reward_kind_name = 'reward.txt'
with open(reward_kind_name, 'r') as f:
    reward_kind = f.read()

filename = ll_path + 'data' + str(rank) + '/'
dot = '.npy'
mac_kind = int(rank_array[0])
sev_kind = int(rank_array[1])

#####################
# VT_1mac_sev_10_80
mac_sev_name = 'VT_' + str(VT_name) + 'mac_sev' + '_' + \
    str(mac_kind) + '_' + str(sev_kind)
mac_trf_name = 'VT_' + str(VT_name) + 'mac_trf' + '_' + \
    str(mac_kind) + '_' + str(sev_kind)

sev_trf_name = 'VT_' + str(VT_name) + 'sev_trf' + \
    str(mac_kind) + '_' + str(sev_kind)
sev_trf_out_name = 'VT_' + str(VT_name) + 'sev_trf_out' + \
    str(mac_kind) + '_' + str(sev_kind)
sev_trf_in_name = 'VT_' + str(VT_name) + 'sev_trf_in' + \
    str(mac_kind) + '_' + str(sev_kind)
sev_cpu_name = 'sev_cpu' + '_' + str(sev_kind) + '_' + str(rank)
sev_mem_name = 'sev_mem' + '_' + str(sev_kind) + '_' + str(rank)

####################
# mac_cpu_name = 'mac_cpu' + '_' + str(mac_kind) + '_' + str(rank)
# mac_mem_name = 'mac_mem' + '_' + str(mac_kind) + '_' + str(rank)

total_cpu_name = 'total_cpu' + '_' + str(mac_kind) + '_' + str(rank)
total_mem_name = 'total_mem' + '_' + str(mac_kind) + '_' + str(rank)

############################
mac_sev_cpu_name = 'VT_' + str(VT_name) + 'mac_sev_cpu' + \
    str(mac_kind) + '_' + str(sev_kind)
mac_sev_mem_name = 'VT_' + str(VT_name) + 'mac_sev_mem' + \
    str(mac_kind) + '_' + str(sev_kind)
now_all_name = 'VT_' + str(VT_name) + 'now_all' + \
    '_' + str(mac_kind) + '_' + str(sev_kind)

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

topk = 0  # np.loadtxt('line2.csv', delimiter=",")
y1 = mac_sev.sum(axis=0) * sev_trf_in
y2 = sev_trf_out * mac_sev.sum(axis=0).reshape((mac_sev.shape[1], 1))
print(np.allclose(y1, y2))
print(np.allclose(y1, sev_trf))

print(trf_cpu_mem.shape)
print(mac_sev_cpu.shape)
print(mac_sev_mem.shape)
print(mac_sev.shape)
print(sev_trf.shape)
print(sev_trf_out.shape)
print(sev_trf_in.shape)

# 参数列表 这里也应该读取文件
para_list = ['actor_lr : ', 'critic_lr : ', 'num_episodes : ', 'gamma : ', 'lmbda : ',
             'epochs : ', 'eps : ', 'deep : ', 'lenth : ',
             'stop_r : ', 'sa : ', 'sb : ', 'epoch_len : ', 'pround : ', 'step_r : ', 'traffic_r : ', 'bs : ']
para_path = '../../../'
with open(para_path + 'para3.txt', 'r', encoding='utf-8') as f:
    para = f.readlines()
    # print(para)
for i, p in enumerate(para):
    para[i] = eval(p.strip(para_list[i] + ',\n'))
    print((para_list[i], para[i]))

actor_lr = para[0]
critic_lr = para[1]
num_episodes = para[2]
gamma = para[3]
lmbda = para[4]
epochs = para[5]
eps = para[6]
deep = para[7]
lenth = para[8]
stop_r = para[9]
s_l = para[10]
s_h = para[11]

step_r = para[14]
traffic_r = para[15]
bs = para[16]
with open(para_path + '../' + 'device.txt', 'r', encoding='utf-8') as f:
    device = torch.device(f.readlines()[0])

env = Envn.Env(
    mac_sev, mac_sev_cpu, mac_sev_mem, trf_cpu_mem, sev_trf, sev_trf_out,
    sev_trf_in, reward_kind, s_cpu, s_mem, deep, lenth, topk, s_l, s_h, traffic_r)
# 随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

model = 'Actor-Critic'
act_dim1 = mac_sev.shape[0]
act_dim2 = mac_sev.shape[1]

agent1 = ActorCritic.ActorCritic(
    act_dim1, actor_lr, critic_lr, gamma, lmbda, epochs, eps, device, model, num_episodes, deep, lenth, bs)
agent2 = ActorCritic.ActorCritic(
    act_dim2, actor_lr, critic_lr, gamma, lmbda, epochs, eps, device, model, num_episodes, deep, lenth, bs)
agent3 = ActorCritic.ActorCritic(
    act_dim1, actor_lr, critic_lr, gamma, lmbda, epochs, eps, device, model, num_episodes, deep, lenth, bs)


all_done = False
punish_list = np.zeros(num_episodes)
reward_list = np.zeros(num_episodes)
dids_list = np.zeros(num_episodes)
return_list = np.zeros(num_episodes)
trf_list = np.zeros((num_episodes, env.mac_sev.shape[0]))
cpus = np.zeros((num_episodes, env.mac_sev.shape[0]))
mems = np.zeros((num_episodes, env.mac_sev.shape[0]))
stop_list = np.zeros((num_episodes, env.mac_sev.shape[0]))
mac_sev_list = np.zeros((num_episodes, mac_sev.shape[0], mac_sev.shape[1]))
epoch_len = para[12]
per_len = int(num_episodes / epoch_len)
round = int(sum(sum(mac_sev)) / para[13])

print('num_episodes, actor_lr, critic_lr, gamma, lmbda, epochs, deep, lenth, round, reward, epoch_len, stop_r, eps, sl, sh step_r, traffic_r bs')
print(num_episodes, actor_lr, critic_lr,
      gamma, lmbda, epochs, deep, lenth, (para[13], round), reward_kind, epoch_len, stop_r, eps, s_l, s_h, step_r, traffic_r, bs)

# print(per_len)
b_stop = 0
b_stop_trf = 0
b_trf = 0
b_trf_stop = 0
cnt = 0
for i in range(epoch_len):
    with tqdm(total=per_len, desc='Iteration %d' % i) as pbar:
        for i_episode in range(per_len):
            start = time.perf_counter()
            # episode_return = 0
            transition_dict = {
                'q1': [],
                'q2': [],
                'q3': [],
                'q1_': [],
                'q2_': [],
                'q3_': [],
                'act_mac1s': [],
                'act_sevs': [],
                'act_mac2s': [],
                'rewards': [],
                'dids': [],
                'dones': [],
                'sev': [],
                'mac1': [],
                'mac2': [],
                'mac_sev': [],
                'pi_1': [],
                'pi_2': [],
                'pi_3': []
            }
            state = env.reset()
            done = False
            run = 0
            while not done:
                # mask mac1
                mask_mac1 = env.which_mac_can_choose_mask()

                action1, q_value1, pi_1 = agent1.take_action(
                    state, mask_mac1, True, env, 1)

                # 选了mac1之后 选他上面的sev 有mask
                mask_sev = env.which_sev_can_choose_mask(action1)

                action2, q_value2, pi_2 = agent2.take_action(
                    state, mask_sev, True, env, 2)

                env.the_now_mac_sev_3agent(action1, action2)

                # 选mac2 有mask
                mask_mac2 = env.mac_2_mask()

                action3, q_value3, pi_3 = agent3.take_action(
                    state, mask_mac2, True, env, 3)

                next_state, reward, did, done = env.step(
                    action3, round)

                tmp = torch.tensor(
                    next_state, dtype=torch.float).to(device).clone()
                tmp = torch.unsqueeze(tmp, 0).to(device)
                q_value1_ = agent1.critic(tmp)
                q_value2_ = agent2.critic(tmp)
                q_value3_ = agent3.critic(tmp)

                # transition_dict['states'].append(np.array(state))
                transition_dict['q1'].append(q_value1)
                transition_dict['q2'].append(q_value2)
                transition_dict['q3'].append(q_value3)

                transition_dict['q1_'].append(q_value1_)
                transition_dict['q2_'].append(q_value2_)
                transition_dict['q3_'].append(q_value3_)

                transition_dict['pi_1'].append(list(pi_1[0]))
                transition_dict['pi_2'].append(list(pi_2[0]))
                transition_dict['pi_3'].append(list(pi_3[0]))

                transition_dict['act_mac1s'].append(int(action1))
                transition_dict['act_sevs'].append(int(action2))
                transition_dict['act_mac2s'].append(int(action3))

                transition_dict['dids'].append(did)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                transition_dict['mac1'].append(mask_mac1)
                transition_dict['sev'].append(mask_sev)
                transition_dict['mac2'].append(mask_mac2)
                transition_dict['mac_sev'].append(env.mac_sev)
                state = next_state

            # 输出对不对
            # rl_utils.TEST_INSTANCE(env, mac_sev, s_cpu, s_mem, transition_dict)
            #########################################
            # 测试本次调度对不对 主要是用mac_sev的值和s_cpu s_mem
            print('步数对不对 dids states next')
            print(len(transition_dict['dids']) == round)
            print(len(transition_dict['q1']) == round)
            print(len(transition_dict['q1_']) == round)
            print(len(transition_dict['q2']) == round)
            print(len(transition_dict['q2_']) == round)
            print(len(transition_dict['q3']) == round)
            print(len(transition_dict['q3_']) == round)
            # print(np.allclose(env.mac_sev.sum(axis=0), mac_sev.sum(axis=0)))
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
            dids_list[cnt] = sum(transition_dict['dids'])
            stop_list[cnt] = env.stop
            # 加入cpu
            cpus[cnt] = env.cpus
            # 加入mems
            mems[cnt] = env.mems
            # 就是一次只存now——trf 然后读到文件里 最后再从文件一个一个读出来 可以不用存整个矩阵了
            trf_list[cnt] = env.now_trf

            if sum(env.stop) > b_stop or (sum(env.stop) == b_stop and sum(env.now_trf) > b_stop_trf):
                torch.save(agent1.actor.state_dict(),
                        './model1/a3_1a_bt_under_bs.pt')
                torch.save(agent1.critic.state_dict(),
                        './model1/a3_1c_bt_under_bs.pt')
                torch.save(agent2.actor.state_dict(),
                        './model1/a3_2a_bt_under_bs.pt')
                torch.save(agent2.critic.state_dict(),
                        './model1/a3_2c_bt_under_bs.pt')
                torch.save(agent3.actor.state_dict(),
                        './model1/a3_3a_bt_under_bs.pt')
                torch.save(agent3.critic.state_dict(),
                        './model1/a3_3c_bt_under_bs.pt')

                b_stop = sum(env.stop)
                b_stop_trf = sum(env.now_trf)

                for tmp in range(0, len(transition_dict['mac_sev']), 50):
                    np.savetxt('./BS_state_change_a3_1/BS_' + str(tmp) + '.csv',
                            transition_dict['mac_sev'][tmp], delimiter=',')
                np.savetxt('./BS_state_change_a3_1/BS_last' + '.csv',
                        transition_dict['mac_sev'][-1], delimiter=',')

            if sum(env.now_trf) > b_trf or (sum(env.stop) > b_stop and sum(env.now_trf) == b_stop_trf):
                torch.save(agent1.actor.state_dict(),
                        './model1/a3_1a_bs_under_bt.pt')
                torch.save(agent1.critic.state_dict(),
                        './model1/a3_1c_bs_under_bt.pt')
                torch.save(agent2.actor.state_dict(),
                        './model1/a3_2a_bs_under_bt.pt')
                torch.save(agent2.critic.state_dict(),
                        './model1/a3_2c_bs_under_bt.pt')
                torch.save(agent3.actor.state_dict(),
                        './model1/a3_3a_bs_under_bt.pt')
                torch.save(agent3.critic.state_dict(),
                        './model1/a3_3c_bs_under_bt.pt')

                b_trf = sum(env.now_trf)
                b_trf_stop = sum(env.stop)

            com_reward, punish = agent1.update(transition_dict, 'mac', sum(stop_list[cnt]), stop_r,
                                            reward_kind, step_r, env.stop, list(env.hcpus), list(env.hmems))
            com_reward, punish = agent2.update(transition_dict, 'which', sum(stop_list[cnt]), stop_r,
                                            reward_kind, step_r, env.stop, list(env.hcpus), list(env.hmems))
            com_reward, punish = agent3.update(transition_dict, 'where', sum(stop_list[cnt]), stop_r,
                                            reward_kind, step_r, env.stop, list(env.hcpus), list(env.hmems))
            # print(sum(com_reward))
            return_list[cnt] = sum(com_reward)
            # print(return_list)
            mac_sev_list[cnt] = env.mac_sev
            punish_list[cnt] = punish
            np.savetxt('./last_state_a3_1/' + str(cnt) + 'last' +
                    '.csv', env.mac_sev, delimiter=',')
            transition_dict.clear()
            del transition_dict
            cnt += 1
            if (i_episode + 1) % per_len == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (per_len * i + i_episode + 1),
                    'return':
                    # 这个输出稍微有点小问题
                    '%.3f' % np.mean(return_list[cnt + 1 - per_len: cnt + 1])
                })
            pbar.update(1)
    
        
        '''
        end = time.perf_counter()
        # 计算运行时间
        runTime = end - start
        f = open('runtime3_1.txt', 'a')
        f.writelines('\n' + str((i * per_len + i_episode + 1, runTime)) + '\n')
        f.close()
        '''

print(dids_list)
print(punish_list)
rl_utils.show(env, stop_list, trf_list, dids_list, mac_kind, sev_kind,
              mac_sev_list, cpus, mems, return_list, rank, reward_kind, 31, para[13], 'para3', punish_list)
del stop_list
del trf_list
del dids_list
del punish_list
del mac_sev_list
del return_list
del cpus
del mems
