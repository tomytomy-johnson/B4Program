import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict, deque
import numpy as np
#from common.gridworld import GridWorld
from common.gridworld_CliffWalking import GridWorld
from common.utils import greedy_probs
from common.utils import softmax_probs
from common.utils import plot_total_reward
from common.utils import smoothing_history, plot_histories, log10_emotion_dict

from ch06.MCTS_T import MCTS_T

class SarsaAgent:
    def __init__(self, gamma=1.0, alpha=0.2, policy='greedy', policy_parameter=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.policy = policy
        self.policy_parameter = policy_parameter
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

        # Emotion
        self.Joy = defaultdict(lambda: 0)
        self.Distress = defaultdict(lambda: 0)
        self.Hope = defaultdict(lambda: 0)
        self.Fear = defaultdict(lambda: 0)

    def get_action(self, state, probs=None):
        if probs != None:
            action_probs = probs
        else:
            action_probs = self.pi[state]
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q = 0 if done else self.Q[next_state, next_action]

        target = reward + self.gamma * next_q
        TD = target - self.Q[state, action]
        q_old = self.Q[state, action]
        self.Q[state, action] += TD * self.alpha
        if self.policy == 'greedy':
            self.pi[state] = greedy_probs(self.Q, state, self.policy_parameter)
        elif self.policy == 'softmax':
            self.pi[state] = softmax_probs(self.Q, state, self.policy_parameter)

        if(TD < 0):
            self.Distress[next_state] = TD
        elif(TD > 0):
            self.Joy[next_state] = TD
        if done:
            pass
    
    def get_Joy(self, state):
        return self.Joy[state]

    def get_Distress(self, state):
        return self.Distress[state]


if __name__ == '__main__':
    env = GridWorld()
    agent = SarsaAgent()
    reward_history = []
    joy_history = []
    distress_history = []
    fear_history = []
    hope_history = []

    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        total_reward = 0
        total_joy = 0
        total_distress = 0
        total_fear = 0
        total_hope = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, done)
            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state
            total_reward += reward
        
        reward_history.append(total_reward)
        if episode % 10 == 0:
            print(episode)
    
    # 情動計算
    Fears = []
    Fear_end_states = defaultdict(lambda: 0)
    Hopes = []
    Hope_end_states = defaultdict(lambda: 0)
    conditions = [{'depth_max': 2, 'loop_N': 100}, {'depth_max': 4, 'loop_N': 15}, {'depth_max': 4, 'loop_N': 100}]
    #conditions = [{'depth_max': 4, 'loop_N': 15}]
    for j in range(len(conditions)):
        d_max, loop = conditions[j]['depth_max'], conditions[j]['loop_N']
        Hope = defaultdict(lambda: 0)
        Fear = defaultdict(lambda: 0)
        print("calculate depth_max: {}, loop_N: {}".format(conditions[j]['depth_max'], conditions[j]['loop_N']))
        for n in range(100):
            size = env.reward_map.shape
            states = [(y, x) for y in range(size[0])
                      for x in range(size[1]) if not env.check_done((y, x))]
            for s in states:
                mcts = MCTS_T(env, agent.Q, s, max_depth=d_max, loop=loop)
                mcts.calculate_Emotion(agent.pi, agent.gamma)
                end_state_f, fear = mcts.fear()
                end_state_h, hope = mcts.hope()
                Fear[s] += (fear - Fear[s]) / (n + 1)
                Hope[s] += (hope - Hope[s]) / (n + 1)
                end_state_f_str = str(end_state_f[0]) if len(end_state_f) == 1 else str(end_state_f[0]) + ',[' + str(len(end_state_f)-1)+']'
                end_state_h_str = str(end_state_h[0]) if len(end_state_h) == 1 else str(end_state_h[0]) + ',[' + str(len(end_state_h)-1)+']'
                Fear_end_states[s] = str(end_state_f_str)
                Hope_end_states[s] = str(end_state_h_str)
                #if s == (2,3):
                #    mcts.print_trajectory(end_state_f[0])
            if n % 10 == 0:
                print(n)
        Fears.append(Fear)
        Hopes.append(Hope)

    #env.render_q(agent.Q)
    #plot_total_reward(smoothing_history(reward_history), ylabel="Total Rewards")
    #plot_total_reward(reward_history, ylabel="Total Rewards")

    #labels = ["Hope", "Fear"]
    #histories = []
    #histories.append(smoothing_history(hope_history))
    #histories.append(smoothing_history(fear_history))
    #plot_histories(histories, labels)

    env.render_e(Fears[0])
    env.render_end_state(Fear_end_states)
    fs = []
    size = env.reward_map.shape
    for f in Fears:
        e = log10_emotion_dict(f)
        fs.append([e[(i, 4)] for i in range(size[0]-2, -1, -1)])
    labels = [ "dmax={}, N={}".format(condition['depth_max'], condition['loop_N']) for condition in conditions]
    xticks = [i for i in range(size[0]-1 -1, -1, -1)]
    plot_histories(fs, labels=labels, xlabel="distance from cliff", ylabel="fear (log10)", xticks_invert=True, xticks=xticks)

    env.render_e(Hopes[0])
    env.render_end_state(Hope_end_states)
    hs = []
    for h in Hopes:
        e = log10_emotion_dict(h)
        hs.append([e[(i, 4)] for i in range(size[0]-2, -1, -1)])
    plot_histories(hs, labels=labels, xlabel="distance from cliff", ylabel="Hope (log10)", xticks_invert=True, xticks=xticks)
