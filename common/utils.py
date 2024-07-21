from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import math as math
from collections import defaultdict


def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


def argmin(xs):
    idxes = [i for i, x in enumerate(xs) if x == min(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


def argmaxs(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))
    # 複数でも返す
    return idxes

#def greedy_probs(Q, state, epsilon=0, action_size=4):
#    qs = [Q[(state, action)] for action in range(action_size)]
#    max_action = argmax(qs)  # OR np.argmax(qs)
#    base_prob = epsilon / action_size
#    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
#    action_probs[max_action] += (1 - epsilon)
#    return action_probs


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = argmaxs(qs)  # OR np.argmax(qs)
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    if type(max_action) != int :
        for action in max_action:
            action_probs[action] += (1 - epsilon) / len(max_action)
    else:
        action_probs[max_action] += (1 - epsilon)
    return action_probs


def softmax_probs(Q, state, temperature=10, action_size=4):
    # 温度が0なら一様分布(ランダム選択になる)を返す
    if(temperature == 0):
        return greedy_probs(Q, state, epsilon=1, action_size=4)

    Qa_probs = []
    Qa_sum = 0.0
    for action in range(action_size):
        Qa_probs.append(math.exp(Q[(state, action)]/temperature))
        Qa_sum += Qa_probs[action]
    action_probs = {action: Qa_probs[action] / Qa_sum for action in range(action_size)}
    return action_probs

def plot_total_reward(reward_history, xlabel="Episodes",ylabel="Total"):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()


def plot_histories(histories, labels, color=None, xlabel="Episodes", ylabel="Total", xticks_invert = False, xticks=None):
    colorlist = ["r", "b", "c", "m", "y", "k", "w"] if color == None else color
    fig, ax = plt.subplots()
    if xticks_invert:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(len(histories)):
        ax.plot(range(len(histories[i])), histories[i], label=labels[i], color=colorlist[i])
    if xticks != None:
        ax.set_xticks(xticks)
    plt.legend()
    plt.show()


def smoothing_history(history, num=10):
    average = []
    for i in range(len(history) - num):
        x = sum(history[i:i+num]) / num
        average.append(x)
    return average

def log10_emotion_dict(emotion_dict):
    e = emotion_dict
    d = defaultdict(lambda: 0)
    for key in e.keys():
        v = e[key]
        if v < 0:
            v = -v
        d[key] = math.log10(v) if v != 0 else 0
    return d
