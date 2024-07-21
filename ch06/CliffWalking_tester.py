import time
from MCTS_T import MCTS_T
from common.utils import plot_total_reward, plot_histories
from common.utils import smoothing_history
from common.utils import softmax_probs
from common.utils import greedy_probs
from sarsa_CliffWalking import SarsaAgent
from q_learning_CliffWalking import QLearningAgent
from common.gridworld_CliffWalking import GridWorld
import numpy as np
from collections import defaultdict
from lib2to3.pytree import convert
import os
import sys
import csv
from random import sample
from tkinter import Grid
# for importing the parent dirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from common.gridworld import GridWorld


class CliffWalking_tester:
    def __init__(self, agent_type='QLearning', episodes=500, gamma=0.9, alpha=0.1, policy='greedy', policy_parameter=0.1):
        self.agent_type = agent_type
        self.episodes = episodes
        self.reward_history = None
        self.env = GridWorld()
        self.agent = None
        self.gamma = gamma
        self.alpha = alpha
        self.policy = policy
        self.policy_parameter = policy_parameter

        self.Qsize = self.env.reward_map.shape
        self.Qave = None
        self.Qfear = None
        self.Qhope = None

    def run(self, sample_num=1):
        self.reward_history = defaultdict(lambda: 0)
        self.Qave = defaultdict(lambda: 0)
        reward_history_sum = [0]*self.episodes

        # 平均導出用ループ
        for loop in range(sample_num):
            if self.agent_type == 'QLearning':
                self.agent = QLearningAgent(self.gamma, self.alpha, self.policy, self.policy_parameter)
            elif self.agent_type == 'Sarsa':
                self.agent = SarsaAgent(self.gamma, self.alpha, self.policy, self.policy_parameter)
            # episode = 試行回数
            for episode in range(self.episodes):
                state = self.env.reset()
                total_reward = 0
                if self.agent_type == 'Sarsa':
                    self.agent.reset()
                    # QLearningはなし

                while True:
                    action = self.agent.get_action(state)
                    next_state, reward, done = self.env.step(action)
                    if self.agent_type == 'QLearning':
                        self.agent.update(state, action, reward, next_state, done)
                    elif self.agent_type == 'Sarsa':
                        self.agent.update(state, action, reward, done)

                    if done:
                        if self.agent_type == 'Sarsa':
                            self.agent.update(next_state, None, None, None)
                            # QLearningはなし
                        break
                    state = next_state
                    total_reward += reward

                reward_history_sum[episode] += total_reward

            states = [(y, x) for y in range(self.Qsize[0])
                      for x in range(self.Qsize[1])]
            for state in states:
                for action in range(len(self.env.action_space)):
                    self.Qave[state, action] += (self.agent.Q[state,action] - self.Qave[state, action]) / (loop + 1)

            if loop % 5 == 0:
                print(loop)

        self.reward_history = [reward_history_sum[episode] /
                               sample_num for episode in range(self.episodes)]

    def run_added_emotion(self, sample_num=1, depth_max=3, loop_N=25, emotion='fear'):
        self.reward_history = defaultdict(lambda: 0)
        self.Qave = defaultdict(lambda: 0)
        reward_history_sum = [0]*self.episodes

        conditions = {'depth_max': depth_max, 'loop_N': loop_N}
        d_max, N = conditions['depth_max'], conditions['loop_N']
        print("calculate depth_max: {}, loop_N: {}".format(d_max, N))

        # 平均導出用ループ
        for loop in range(sample_num):
            print('sample: {}'.format(loop))
            if self.agent_type == 'QLearning':
                self.agent = QLearningAgent(self.gamma, self.alpha, self.policy, self.policy_parameter)
            elif self.agent_type == 'Sarsa':
                self.agent = SarsaAgent(self.gamma, self.alpha, self.policy, self.policy_parameter)
            # episode = 試行回数
            for episode in range(self.episodes):
                if episode % 10 == 0:
                    print(' |- episode: {}'.format(episode))
                state = self.env.reset()
                total_reward = 0
                if self.agent_type == 'Sarsa':
                    self.agent.reset()
                    # QLearningはなし

                while True:
                    # 情動計算
                    mcts = MCTS_T(self.env, self.agent.Q, state, max_depth=d_max, loop=N)
                    pi = self.agent.b if self.agent_type == 'QLearning' else self.agent.pi
                    mcts.calculate_Emotion(pi, self.agent.gamma)
                    if emotion == 'fear':
                        end_states, emotion_value = mcts.fear()
                    elif emotion == 'hope':
                        end_states, emotion_value = mcts.hope()
                    # emotionを考慮したQ値
                    bias = self.agent.gamma
                    actions_count = mcts.trajectory_start_action(end_states)
                    for i in range(len(self.env.action_space)):
                        self.agent.Q[state, i] += bias * (emotion_value * actions_count[i] / sum(actions_count))

                    action = self.agent.get_action(state)
                    next_state, reward, done = self.env.step(action)
                    if self.agent_type == 'QLearning':
                        self.agent.update(state, action, reward, next_state, done)
                    elif self.agent_type == 'Sarsa':
                        self.agent.update(state, action, reward, done)

                    if done:
                        if self.agent_type == 'Sarsa':
                            self.agent.update(next_state, None, None, None)
                            # QLearningはなし
                        break
                    state = next_state
                    total_reward += reward

                reward_history_sum[episode] += total_reward
            states = [(y, x) for y in range(self.Qsize[0]) for x in range(self.Qsize[1])]
            for state in states:
                for action in range(len(self.env.action_space)):
                    self.Qave[state, action] += (self.agent.Q[state, action] - self.Qave[state, action]) / (loop + 1)

        self.reward_history = [reward_history_sum[episode] / sample_num for episode in range(self.episodes)]

    def run_emotion_map(self, sample_num=1, depth_max=3, loop_N=25):
        # 通常の学習
        if self.agent_type == 'QLearning':
            self.agent = QLearningAgent(self.gamma, self.alpha, self.policy, self.policy_parameter)
        elif self.agent_type == 'Sarsa':
            self.agent = SarsaAgent(self.gamma, self.alpha, self.policy, self.policy_parameter)

        for episode in range(self.episodes):
            if episode % 10 == 0:
                print(' |- episode: {}'.format(episode))
            state = self.env.reset()
            if self.agent_type == 'Sarsa':
                self.agent.reset()
                # QLearningはなし
            while True:
                action = self.agent.get_action(state)
                next_state, reward, done = self.env.step(action)
                if self.agent_type == 'QLearning':
                    self.agent.update(state, action, reward, next_state, done)
                elif self.agent_type == 'Sarsa':
                    self.agent.update(state, action, reward, done)
                if done:
                    if self.agent_type == 'Sarsa':
                        self.agent.update(next_state, None, None, None)
                        # QLearningはなし
                    break
                state = next_state

        self.Qave = defaultdict(lambda: 0)
        self.Qfear = defaultdict(lambda: 0)
        self.Qhope = defaultdict(lambda: 0)

        d_max, N = depth_max, loop_N
        print("calculate depth_max: {}, loop_N: {}".format(d_max, N))

        # 平均導出用ループ
        for loop in range(sample_num):
            print('sample: {}'.format(loop))
            states = [(y, x) for y in range(self.Qsize[0]) for x in range(self.Qsize[1]) if not self.env.check_done((y, x))]
            for state in states:
                # 情動計算
                mcts = MCTS_T(self.env, self.agent.Q, state, max_depth=d_max, loop=N)
                pi = self.agent.b if self.agent_type == 'QLearning' else self.agent.pi
                mcts.calculate_Emotion(pi, self.agent.gamma)
                end_states_f, fear = mcts.fear()
                end_states_h, hope = mcts.hope()
                self.Qfear[state] += (fear - self.Qfear[state]) / (loop + 1)
                self.Qhope[state] += (hope - self.Qhope[state]) / (loop + 1)
                #end_states_f_str = str(end_states_f[0]) if len(end_states_f) == 1 else str(end_states_f[0]) + ',[' + str(len(end_states_f)-1)+']'
                #end_states_h_str = str(end_states_h[0]) if len(end_states_h) == 1 else str(end_states_h[0]) + ',[' + str(len(end_states_h)-1)+']'
                #Fear_end_states[s] = str(end_states_f_str)
                #Hope_end_states[s] = str(end_states_h_str)


if __name__ == '__main__':
    types = ['QLearning', 'Sarsa']
    agent_type, policy = types[1], 'greedy'
    episodes = 500
    sample_num = 10
    depth_max , loop_N = 3, 25
    gamma = 1.0
    alpha = 0.2
    policy_parameters = {'greedy': 0.10, 'softmax': 1.0}
    tester = CliffWalking_tester(agent_type, episodes, gamma, alpha, policy, policy_parameters[policy])
    Qs = []

    reward_histories = []

    # Hope slip (d, N)
    d_N = [(1, 10), (1, 50), (3, 10), (3, 50)]
    tester.agent_type = agent_type
    print(tester.agent_type + ' hope')
    labels = []
    for item in d_N:
        d, N = item
        tester.run_added_emotion(sample_num, d, N, emotion='hope')
        reward_histories.append(tester.reward_history)
        Qs.append(tester.Qave)
        labels.append('dmax=' + str(d) + ', N=' + str(N))
    histories = [smoothing_history(history) for history in reward_histories]
    plot_histories(histories, labels)

    filename = agent_type + '_Reward_hope.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = labels
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(episodes):
            writer.writerow({labels[0]: reward_histories[episode], labels[1]: reward_histories[episode], labels[2]: reward_histories[episode], labels[3]: reward_histories[episode]})


    ## emotion map
    #print(tester.agent_type + ' emotion map')
    #tester.run_emotion_map(sample_num, depth_max, loop_N)
    #tester.env.render_e(tester.Qfear)
    #tester.env.render_e(tester.Qhope)
    

    ## Normal
    #print(tester.agent_type + ' normal')
    #tester.run(sample_num=sample_num)
    #reward_hisotry_normal = tester.reward_history
    #Qs.append(tester.Qave)

    ## Fear
    #tester.agent_type = agent_type
    #print(tester.agent_type + ' fear')
    #tester.run_added_emotion(sample_num, depth_max, loop_N,emotion='fear')
    #reward_hisotry_fear = tester.reward_history
    #Qs.append(tester.Qave)

    ## Hope
    #tester.agent_type = agent_type
    #print(tester.agent_type + ' hope')
    #tester.run_added_emotion(sample_num, depth_max, loop_N,emotion='hope')
    #reward_hisotry_hope = tester.reward_history
    #Qs.append(tester.Qave)

    #start = time.time()
    #end = time.time()
    #print('処理時間:{}'.format(round(end - start, 3)))

    #tester.env.render_q(Qs[0])
    #tester.env.render_q(Qs[1])
    #tester.env.render_q(Qs[2])

    #labels = [agent_type, agent_type + ' Fear', agent_type + ' Hope']
    #colorlist = ['g', 'b', 'r']
    #histories = [smoothing_history(reward_hisotry_normal), smoothing_history(reward_hisotry_fear), smoothing_history(reward_hisotry_hope)]
    #plot_histories(histories, labels, colorlist)

    #filename = agent_type + 'average_10_normal_fear_hope_d3N25.csv'
    #with open(filename, 'w', newline='') as csvfile:
    #    fieldnames = labels
    #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #    writer.writeheader()
    #    for episode in range(episodes):
    #        writer.writerow({labels[0]: reward_hisotry_normal[episode], labels[1]: reward_hisotry_fear[episode], labels[2]: reward_hisotry_hope[episode]})
