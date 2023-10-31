import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import copy
import collections
from modules import rl_utils
import time
import scipy.stats as stats
# 然后把该输出的都输出一遍


class Env:
    def __init__(self, mac_sev, mac_sev_cpu, mac_sev_mem, now_total, sev_trf,
                 sev_trf_out, sev_trf_in, reward, s_cpu, s_mem, deep, lenth, topk, sl, sh, p_traffic):
        self.now = now_total
        self.ini = copy.deepcopy(self.now)
        self.mac_index = np.arange(0, mac_sev.shape[0])
        # self.now要随着更改的 判断一下self.now跟着改没改 单值不会改 对象会改
        # now_total: 0 mac_trf; 1-3 now_cpu, total_cpu, cpu% 4-6 now_mem, total_mem, mem
        self.hcpus = self.now[:, 2]  # total cpu
        self.hmems = self.now[:, 5]  # total mem
        self.init_cpus = self.ini[:, 1]  # ini_now_cpu
        self.init_mems = self.ini[:, 4]  # ini_now_mem
        self.cpus = self.now[:, 1]  # now_cpu
        self.mems = self.now[:, 4]  # now_mem

        self.sev_cpu = s_cpu  # sev_cpu
        self.sev_mem = s_mem  # sev_mem

        self.mac_sev_cpu = mac_sev_cpu
        self.mac_sev_mem = mac_sev_mem

        self.leftcpus = self.hcpus - self.cpus  # now_left_cpu
        self.leftmems = self.hmems - self.mems  # now_left_mem
        self.pcpu = self.now[:, 3]
        self.pmem = self.now[:, 6]

        self.mac_sev = mac_sev  # 40 * 941 元素是服务个数
        self.ini_sev = copy.deepcopy(mac_sev)  # 初始分布 以便reset
        self.now_trf = self.now[:, 0]  # 当前每台物理机流量
        self.ini_trf = copy.deepcopy(
            now_total[:, 0])  # 做个备份 初始时每台物理机流量 以便reset

        self.sev_trf = sev_trf
        self.sev_trf_out = sev_trf_out
        self.sev_trf_in = sev_trf_in

        self.mask = np.ones(self.mac_sev.shape[0])  # self.now[:, -1]
        self.stop = self.now[:, -1]  # np.zeros(self.mac_sev.shape[0]) #目前都没关停

        self.mac_sev_list = []
        self.rnd = -1

        self.m = 0  # 当前是第几个mac
        self.s = 0  # 当前是不为0的第几个sev
        self.cnt = 0  # 一个计数器
        self.s_now = 0  # 这个s_now是不为零的某个sev的index 为了编程实现定义的变量
        self.reward = reward  # 选择不同的Reward

        self.total_steps = 0
        self.mac_mask = np.ones(self.mac_sev.shape[0])
        self.sev_mask = np.ones(self.mac_sev.shape[1])
        self.a_2_mask = np.ones(self.mac_sev.shape[0] * self.mac_sev.shape[1])

        self.ms_num = deep
        self.lenth = lenth
        self.topk = topk
        self.p = 0.5
        self.sl = sl
        self.sh = sh
        self.p_traffic = p_traffic
    '''
    for:
        state 判断是否done
        policy预测act 通过选择做出act
        env给出act合法性 给reward 到next state
        value 给出打分
    end 
    '''

    def store_state(self, mac_sev):
        self.mac_sev_list.append(mac_sev)
        self.rnd += 1  # 当前mac_sev_index

    def pre_handle_mat(self):

        # 这个每次手动改一下
        s1 = self.sl
        s2 = self.sh

        # reshape 然后padding
        pmac_sev = copy.deepcopy(self.mac_sev).reshape((s1, s2))
        pmac_sev_cpu = copy.deepcopy(self.mac_sev_cpu).reshape((s1, s2))
        pmac_sev_mem = copy.deepcopy(self.mac_sev_mem).reshape((s1, s2))

        return pmac_sev, pmac_sev_cpu, pmac_sev_mem

    def Reshape_And_Padding(self):
        # 主要进行padding
        '''
        mac_sev 40 * 941 -> 256 * 256
        now_total 40 * 7 -> 256 * 256 这个之后可以加点东西 比如加上当前是mac sev
        mac_sev_cpu 40 * 941 -> 256 * 256
        mac_sev_mem 40 * 941 -> 256 * 256
        mac_sev_pos 40 * 941 -> 256 * 256 在当前mac sev上写1 其余为0
        '''
        # reshape  160 * 236 -> 256 * 256

        '''
        mac_sev : 5 * 100 -> 20 * 25 -> 28 * 28 
        mac_sev_cpu/mem 5 * 100 -> 20 * 25
        sev_trf : 100 * 100 -> ?这个着实不好让他知道 但是state需要知道 100 * 99 / 2 = 99 * 50 = 2 * 50 * 50 
        -> 2 * 25 * 2 * 25 * 2 = 32 * 32
        '''

        mac_sev, mac_sev_cpu, mac_sev_mem = self.pre_handle_mat()

        lenth = self.lenth  # image边长

        s1 = int((lenth - mac_sev.shape[0]) / 2)
        s2 = lenth - mac_sev.shape[0] - s1
        s3 = int((lenth - mac_sev.shape[1]) / 2)
        s4 = lenth - mac_sev.shape[1] - s3
        mac_sev = np.pad(mac_sev, ((s1, s2), (s3, s4)))
        mac_sev_cpu = np.pad(mac_sev_cpu, ((s1, s2), (s3, s4)))
        mac_sev_mem = np.pad(mac_sev_mem, ((s1, s2), (s3, s4)))

        # reshape 5 * 8 -> 28 * 28 这里注意cpu% 更新 8列 有一列是stop
        sa = int((lenth - self.now.shape[0]) / 2)
        sb = lenth - self.now.shape[0] - sa
        sc = int((lenth - self.now.shape[1]) / 2)
        sd = lenth - self.now.shape[1] - sc
        now_total = np.pad(self.now, ((sa, sb), (sc, sd)))

        # 多智能体的话这个需要改一下
        # mac_sev_pos mac,sev:1 40 * 941
        mac_sev_pos = np.zeros((self.mac_sev.shape[0], self.mac_sev.shape[1]))
        mac_sev_pos[self.m][self.s_now] = 1
        # mac_sev_pos = np.pad(mac_sev_pos, ((0, 0), (0, 3)))
        # mac_sev_pos = mac_sev_pos.reshape((20, 25))
        # mac_sev_pos = np.pad(mac_sev_pos, ((s1, s2), (s3, s4)))

        return now_total, mac_sev, mac_sev_cpu, mac_sev_mem, mac_sev_pos

    def YesorNo(self):  # 自己尽量不要mask 因为如果都是迁移 有些反而不能迁但是物理上符合迁移
        Yes_Or_No = (((self.hcpus - self.cpus) >= self.sev_cpu[self.s_now]).astype(int)
                     * ((self.hmems - self.mems) >= self.sev_mem[self.s_now]).astype(int)
                     * (self.stop == 0).astype(int))
        return Yes_Or_No

    def ReYesorNo(self):
        Re_Yes_Or_No = (((self.hcpus - self.cpus) >= self.sev_cpu[self.s_now]).astype(int)
                        * ((self.hmems - self.mems) >= self.sev_mem[self.s_now]).astype(int)
                        * (self.stop == 0).astype(int)
                        * (self.mac_index != self.m).astype(int))
        return Re_Yes_Or_No

    def CosineDistance(self, x, y):
        x = np.array(x)
        y = np.array(y)
        s = (np.linalg.norm(x)*np.linalg.norm(y))
        if s == 0:
            return 0
        return np.dot(x, y)/s

    # 3agent/1agent/2agent mac2 mask
    def mac_2_mask(self):
        self.mask = (
            np.ones(self.mac_sev.shape[0]) * self.YesorNo()).astype(int)
        # for m in range(self.mac_sev.shape[0]):
        # self.mask[m] = self.YesorNo(m)
        return self.mask

    # MARL 3agent
    def the_now_mac_sev_3agent(self, mac, sev):
        # act_mac_sev : 0 - 499
        # index 99 199
        self.m = mac
        self.s_now = sev

    # MARL 3agent has mask or no mask 当前能选哪一个物理机 其实就是没关的物理机 可以从里面迁移出来
    def which_mac_can_choose_mask(self):
        # mac
        mac_now = self.mac_sev.sum(axis=1)
        self.mac_mask = (mac_now > 0).astype(int)
        # self.mac_mask = np.ones(self.mac_sev.shape[0])
        # for m in range(self.mac_sev.shape[0]):
        #    if sum(self.mac_sev[m]) == 0:
        #        self.mac_mask[m] = 0

        return self.mac_mask

    # MARL 3agent has mask 选了mac1之后 选择mac1上的那个sev
    def which_sev_can_choose_mask(self, mac):
        self.sev_mask = (
            np.ones(self.mac_sev.shape[1]) * (self.mac_sev[mac] > 0).astype(int)).astype(int)
        # for s in range(self.mac_sev.shape[1]):
        #    if self.mac_sev[mac][s] == 0:
        #        self.sev_mask[s] = 0

        return self.sev_mask

    # 1agent
    def the_now_mac_sev_pos_1agent(self):
        # 一个计数器 已到当前服务 先+1
        self.cnt += 1
        # 从初始mac_sev里遍历 先行再列 选出ini_sev里不是0的sev的index 放在sevs里
        sevs = [i for i in range(self.mac_sev.shape[1])
                if self.ini_sev[self.m][i] != 0]
        # self.s是当前sev在sevs里的序号 self.s_now是sev在mac_sev里的序号
        self.s_now = sevs[self.s]

    # 顺序遍历时候的做法#1agent
    def the_next_mac_sev_pos_1agent(self):
        done = False
        # 判断下一次是迁移哪一个
        if (self.cnt >= self.ini_sev[self.m][self.s_now]):
            self.s += 1
            self.cnt = 0
            sevs = [i for i in range(self.mac_sev.shape[1])
                    if self.ini_sev[self.m][i] != 0]
            if (self.s < len(sevs)):
                sevs = [i for i in range(
                    self.mac_sev.shape[1]) if self.ini_sev[self.m][i] != 0]
                self.s_now = sevs[self.s]
            else:
                self.m += 1
                self.s = 0
                if (self.m < self.ini_sev.shape[0]):
                    sevs = [i for i in range(
                        self.mac_sev.shape[1]) if self.ini_sev[self.m][i] != 0]
                    self.s_now = sevs[self.s]
                else:
                    done = True
                    self.m = 0
        return done

    def step_1a(self, act):
        done = False
        # 第一步会出问题 之前的代码没有找第一步

        # 当前执行哪个服务
        # 然后迁移并更新
        state, did, inc, dec, cos1_2, cos2_2 = self.get_new_state(
            act)
        reward = self.give_reward(
            act, did, inc, dec, 0, cos1_2, cos2_2)

        done = self.the_next_mac_sev_pos_1agent()

        return state, reward, did, done

    def step(self, act, round):
        done = False
        # 第一步会出问题 之前的代码没有找第一步
        # 当前执行哪个服务

        # 然后迁移并更新
        state, did, inc, dec, cos1_2, cos2_2 = self.get_new_state(
            act)
        reward = self.give_reward(
            act, did, inc, dec, round, cos1_2, cos2_2)

        self.total_steps += 1
        if self.total_steps == round:
            done = True

        return state, reward, did, done

    # trf也是一种State了

    def get_the_traffic(self, mac):
        '''
        mac_sevs_index = [i for i in range(
            self.mac_sev.shape[1]) if self.mac_sev[mac][i] != 0]
        now_sevs_index = [i for i in range(
            self.mac_sev.shape[1]) if self.mac_sev[self.m][i] != 0]
        '''
        # self.s_now
        # self.mac_sev是没变的
        mac_sev_clone = copy.deepcopy(self.mac_sev)
        mac_sev_clone[self.m][self.s_now] -= 1
        mac_sev_clone[mac][self.s_now] += 1
        # 判断你当前迁移的服务与其他服务的流量 对迁移到的物理机的增量 对迁移走的物理机的减小
        trf1 = mac_sev_clone[mac] * self.sev_trf_in
        tmp = copy.deepcopy(mac_sev_clone[mac].reshape(
            (self.mac_sev.shape[1], 1)))
        trf2 = tmp * self.sev_trf_out
        mat_mac_trf = (trf1 < trf2).astype(int) * trf1 + \
            (trf1 >= trf2).astype(int) * trf2

        t1 = mac_sev_clone[self.m] * self.sev_trf_in
        t = copy.deepcopy(mac_sev_clone[self.m].reshape(
            (self.mac_sev.shape[1], 1)))
        t2 = t * self.sev_trf_out
        mat_selfm_trf = (t1 < t2).astype(int) * t1 + \
            (t1 >= t2).astype(int) * t2

        the_mac_trf = mat_mac_trf.sum()
        the_selfm_trf = mat_selfm_trf.sum()
        '''
        for i in range(self.mac_sev.shape[1] - 1):
            for j in range(i + 1, self.mac_sev.shape[1]):
                # 之后self.m的流量
                after_self += min(mac_sev_clone[self.m][i] * self.sev_trf_out[i]
                                  [j], mac_sev_clone[self.m][j] * self.sev_trf_in[i][j])
                # 之后mac的流量
                after_mac += min(mac_sev_clone[mac][i] * self.sev_trf_out[i]
                                 [j], mac_sev_clone[mac][j] * self.sev_trf_in[i][j])
        '''
        increase = the_mac_trf - self.now_trf[mac]
        decrease = self.now_trf[self.m] - the_selfm_trf

        return increase, decrease

    def update_the_trf(self, act):
        inc, dec = self.get_the_traffic(act)
        self.now_trf[act] += inc
        self.now_trf[self.m] -= dec
        return inc, dec

    def get_new_state(self, act):
        mac = act
        did = 0
        inc = 0
        dec = 0
        yes = self.ReYesorNo()
        # yes = yes and (inc - dec > 0)
        # if yes[act] > 0:
        #    inc, dec = self.update_the_trf(act)

        # 如果迁移了把它关了 那这一步应该怎么办
        # 所以我们需要关机 只需要让迁过去的更好即可
        # p1_1 = self.mac_sev[self.m]
        p1_2 = self.mac_sev[act]
        q = self.ini_sev.sum(axis=0)
        # cos1_1 =
        cos1_2 = self.CosineDistance(p1_2, q)

        '''
        # q1_1 = np.stack((q, q), axis=0)
        P1_1 = F.softmax(torch.tensor(p1_1, dtype=float), dim=0)
        P1_2 = F.softmax(torch.tensor(p1_2, dtype=float), dim=0)
        Q1 = F.softmax(torch.tensor(q, dtype=float), dim=0)
        kl1_1 = stats.entropy(P1_1, Q1)
        kl1_2 = stats.entropy(P1_2, Q1)
        '''

        if (yes[act] > 0):
            inc, dec = self.get_the_traffic(act)
        if (yes[act] > 0 and ((inc - dec) > 0)):
            did = 1
            inc, dec = self.update_the_trf(act)

            self.cpus[self.m] -= self.sev_cpu[self.s_now]
            self.mems[self.m] -= self.sev_mem[self.s_now]
            self.cpus[mac] += self.sev_cpu[self.s_now]
            self.mems[mac] += self.sev_mem[self.s_now]

            # 剩余空间 初始空间 mac上sev的个数 cpu mem
            self.leftcpus = self.hcpus - self.cpus
            self.leftmems = self.hmems - self.mems
            self.pcpu = self.cpus / self.hcpus
            self.pmem = self.mems / self.hmems

            # self.s 序号 记录当前选择的是哪个sev 没成功一样记录
            self.mac_sev[self.m][self.s_now] -= 1
            self.mac_sev[mac][self.s_now] += 1
            # self.now[:, -1] = self.mask

            if sum(self.mac_sev[self.m]) == 0:
                self.stop[self.m] = 1
                print(('关了', self.m))
                if self.cpus[self.m] != 0 or self.mems[self.m] != 0:
                    print(self.cpus[self.m])
                    print("False")
                else:
                    print("True")

        # p2_1 = self.mac_sev[self.m]
        p2_2 = self.mac_sev[act]
        q = self.ini_sev.sum(axis=0)
        # q1_1 = np.stack((q, q), axis=0)
        '''
        P2_1 = F.softmax(torch.tensor(p2_1, dtype=float), dim=0)
        P2_2 = F.softmax(torch.tensor(p2_2, dtype=float), dim=0)
        Q1 = F.softmax(torch.tensor(q, dtype=float), dim=0)
        kl2_1 = stats.entropy(P2_1, Q1)
        kl2_2 = stats.entropy(P2_2, Q1)
        '''
        cos2_2 = self.CosineDistance(p2_2, q)

        now_total, mac_sev, mac_sev_cpu, mac_sev_mem, mac_sev_pos = self.Reshape_And_Padding()
        lenth = self.lenth
        now_total = now_total.reshape(1, lenth, lenth)
        mac_sev = mac_sev.reshape(1, lenth, lenth)
        mac_sev_cpu = mac_sev_cpu.reshape(1, lenth, lenth)
        mac_sev_mem = mac_sev_mem.reshape(1, lenth, lenth)

        self.store_state(mac_sev)

        mac_sev_num = self.ms_num
        if len(self.mac_sev_list) >= mac_sev_num:
            some_mac_sev = copy.deepcopy(mac_sev)
            for i in range(1, mac_sev_num):
                some_mac_sev = np.concatenate(
                    (some_mac_sev, self.mac_sev_list[self.rnd - i]), axis=0)
        else:
            some_mac_sev = copy.deepcopy(mac_sev)
            for i in range(self.rnd):
                some_mac_sev = np.concatenate(
                    (some_mac_sev, self.mac_sev_list[i]), axis=0)
            for i in range(mac_sev_num - self.rnd - 1):
                some_mac_sev = np.concatenate(
                    (some_mac_sev, self.mac_sev_list[self.rnd]), axis=0)

        now_total = now_total / now_total.max()
        some_mac_sev = some_mac_sev / some_mac_sev.max()
        mac_sev_cpu = mac_sev_cpu / mac_sev_cpu.max()
        mac_sev_mem = mac_sev_mem / mac_sev_mem.max()
        '''
        state = np.concatenate(
            (now_total, some_mac_sev, mac_sev_cpu, mac_sev_mem), axis=0)'''
        state = some_mac_sev

        return state, did, inc, dec, cos1_2, cos2_2

    def cal_the_rules_score(self, mac, mac_sev):
        score = np.ones(self.topk.shape[0])
        for k in range(self.topk.shape[0]):
            n_list = [n for n in range(
                self.topk.shape[1]) if self.topk[k][n] != -1]
            for n in n_list:
                if mac_sev[mac][n] != 0:
                    score[k] *= mac_sev[mac][n]
        return sum(score)

    def give_reward(self, act, did, inc, dec, round, cos1_2, cos2_2):
        reward = 0
        '''
        stop = 0
        if (sum(self.stop) - sum(stop_l)) != 0:
            stop = 1
        '''

        '''
        score1 = self.cal_the_rules_score(act, pre_mac_sev)
        score2 = self.cal_the_rules_score(act, self.mac_sev)

        score_up = score2 / score1
        '''

        if self.reward == 'r2 * 0':
            reward_tf = torch.sigmoid(torch.tensor(inc - dec)) * did
            reward += reward_tf

        elif self.reward == 'r1':
            reward = 0

        elif (self.reward == 'r2' or self.reward == 'r1 + r2 + r3'
              or self.reward == 'r1 + r2' or self.reward == 'r2 + r3'
              or self.reward == 'r3' or self.reward == 'r1 + r3'):

            reward_tf = torch.sigmoid(torch.tensor(
                inc - dec) * did) * self.p_traffic
            reward += reward_tf

        elif (self.reward == 'cos'):
            reward = (cos2_2 - cos1_2) * did

        return reward

    def reset(self):
        self.now = copy.deepcopy(self.ini)

        self.cpus = copy.deepcopy(self.init_cpus)
        self.mems = copy.deepcopy(self.init_mems)
        self.leftcpus = self.hcpus - self.cpus
        self.leftmems = self.hmems - self.mems
        self.pcpu = self.now[:, 3]
        self.pmem = self.now[:, 6]

        self.mask = np.ones(self.mac_sev.shape[0])  # self.now[:, -1]

        self.now_trf = copy.deepcopy(self.ini_trf)

        self.mac_sev = copy.deepcopy(self.ini_sev)

        self.m = 0
        self.s = 0
        self.s_now = 0
        self.stop = self.now[:, -1]  # np.zeros(self.mac_sev.shape[0])
        self.mac_sev_list = []
        self.rnd = -1
        self.cnt = 0

        self.total_steps = 0
        self.mac_mask = np.ones(self.mac_sev.shape[0])
        self.sev_mask = np.ones(self.mac_sev.shape[1])

        lenth = self.lenth
        now_total, mac_sev, mac_sev_cpu, mac_sev_mem, mac_sev_pos = self.Reshape_And_Padding()

        now_total = now_total.reshape(1, lenth, lenth)
        mac_sev = mac_sev.reshape(1, lenth, lenth)
        mac_sev_cpu = mac_sev_cpu.reshape(1, lenth, lenth)
        mac_sev_mem = mac_sev_mem.reshape(1, lenth, lenth)

        self.store_state(mac_sev)  # 1 * 28 * 28

        mac_sev_num = self.ms_num
        some_mac_sev = copy.deepcopy(mac_sev)
        for i in range(mac_sev_num - 1):
            some_mac_sev = np.concatenate(
                (some_mac_sev, self.mac_sev_list[self.rnd - 1]), axis=0)

        now_total = now_total / now_total.max()
        some_mac_sev = some_mac_sev / some_mac_sev.max()
        mac_sev_cpu = mac_sev_cpu / mac_sev_cpu.max()
        mac_sev_mem = mac_sev_mem / mac_sev_mem.max()

        state = some_mac_sev
        '''state = np.concatenate(
            (now_total, some_mac_sev, mac_sev_cpu, mac_sev_mem), axis=0)'''
        return state
