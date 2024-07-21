from lib2to3.pytree import convert
import os, sys
import csv
from random import sample
from tkinter import Grid; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs

import math
import numpy as np
from collections import defaultdict

from common.gridworld_CliffWalking import GridWorld
from common.utils import argmax, argmin
from ch06.Node import Node

class MCTS_T:
    def __init__(self, env, Q, state, max_depth, loop):
        #print("==============")
        self.env:GridWorld = env
        self.Q = Q
        self.root = Node(state, depth=0, parent_node=None, parent_action=None)
        self.max_depth = max_depth
        self.loop = loop
        self.trajectory_end_nodes = []
        
        self.fears = defaultdict(lambda: 0)
        self.hopes = defaultdict(lambda: 0)

        num = 0
        while num < self.loop:
            #print('MCTS_T start')
            current_node = self.select_child_node(self.root)
            current_node = self.expand_current_node(current_node)
            self.backup(current_node)
            num += 1

        #print("end_node_num:"+str(len(self.trajectory_end_nodes)))
    
    def get_action_space(self, state):
        action_space =  [0, 1, 2, 3]
        return action_space
        y, x = state
        if y == 0:
            # remove UP
            action_space.remove(0)
        if y == self.env.height - 1:
            # remove DOWN
            action_space.remove(1)
        if x == 0:
            # remove LEFT
            action_space.remove(2)
        if x == self.env.width - 1:
            # remove RIGHT
            action_space.remove(3)
        return action_space
    
    def select_child_node(self, node:Node):
        #print('slect_child_node start')
        current = node
        while True:
            # 終端なら返す
            if self.env.check_done(current.state) or current.depth == self.max_depth:
                return current

            # 遷移可能な子ノードが全部追加されているか
            for action in self.get_action_space(current.state):
                next_states = self.env.get_trans_states(current.state, action)
                for next_state in next_states:
                    # 追加されていない子ノードがあればcurrentを返す
                    if current.next_nodes[action, next_state] == None:
                        return current

            current = self.select_child(current)

    def select_child(self, node:Node):
        #print('slect_child start')
        current:Node = node
        # UCT1の値
        values = []
        # max_valueのarg → action用
        exchange = []

        for action in self.get_action_space(current.state):
            next_states = self.env.get_trans_states(current.state, action)
            child_nodes = [current.next_nodes[action, next] for next in next_states]
            # stochastic(複数の状態に遷移している状態)なアクションの不確かさの更新
            #if len(next_states) > 1:
            #    numerators = 0.0
            #    denominators = 0.0
            #    for child in child_nodes:
            #        numerators += child.count_num * child.sigma[child.state]
            #        denominators += child.count_num
            #    current.sigma[current.state, action] = numerators / denominators
            
            numerators = 0.0
            denominators = 0.0
            for child in child_nodes:
                numerators += child.count_num * child.sigma[child.state]
                denominators += child.count_num
            current.sigma[current.state, action] = numerators / denominators
            #print('{},{}: {}'.format(current.state, action,current.sigma[current.state, action]))
            
            # UCT1
            q = self.Q[current.state, action]
            sig = current.sigma[current.state, action]
            count_num_state = current.count_num
            count_num_action = sum([child.count_num for child in child_nodes])

            v = q + sig * math.sqrt((2*math.log(count_num_state)) / count_num_action)
            values.append(v)
            # valueのindex -> actionへ逆引き用
            exchange.append(action)

        select_action = exchange[argmax(values)]
        next_state = self.env.next_state(current.state, select_action)
        return current.next_nodes[select_action, next_state]

    def expand_current_node(self, node:Node):
        #print('expand_current_node start')
        current = node
        # 終了状態もしくは指定深さのノードなら飛ばす
        if self.env.check_done(current.state) or current.depth == self.max_depth:
            return current
        else:
            non_added_child_keys = []
            for action in self.get_action_space(current.state):
                next_states = self.env.get_trans_states(current.state, action)
                non_added_child_keys.extend((action, next) for next in next_states if current.next_nodes[action,next] == None)
            choice = np.random.choice(list(range(len(non_added_child_keys))))
            select_key = non_added_child_keys[choice]
            action, next_state = select_key
            child = Node(state=next_state, depth=current.depth+1, parent_node=current, parent_action=action)
            # childが終端状態もしくは終端ノードならsigma=0, それら以外はsigma=1に設定
            if self.env.check_done(child.state) or child.depth == self.max_depth:
                child.sigma[child.state] = 0
            else:
                child.sigma[child.state] = 1
            current.next_nodes[action, next_state] = child
            self.trajectory_end_nodes.append(child)
            return child

    def backup(self, node):
        #print('backup start')
        current = node
        # rootノードまでバックアップ
        while True:
            current.count_num += 1
            # 葉ノードでないなら不確かさsigmaを更新
            if not self.env.check_done(current.state) and current.depth != self.max_depth:
                ms = []
                sig = []
                for action in self.get_action_space(current.state):
                    next_states = self.env.get_trans_states(current.state, action)
                    for next in next_states:
                        child = current.next_nodes[action, next]
                        # if n(s') >= 1 : m(s') = n(s'), sigma*(s') = sigma(s'), otherwise : m(s') = sigma*(s') = 1
                        if child == None or child.count_num == 0:
                            ms.append(1)
                            sig.append(1)
                        else:
                            ms.append(child.count_num)
                            sig.append(child.sigma[child.state])
                # Update Uncertainty for node s (sigma)
                current.sigma[current.state] = sum([m * s for m, s in zip(ms, sig)]) / sum(ms)
                # print("sigma[" + str(current.state) + "] : " + str(current.sigma[current.state]))
            # 一つ上のノードに上がる
            current = current.parent_node
            # 根ノードであったなら終了
            if current == None:
                return
    
    def print_trajectory(self, end_state):
        end_node = [node for node in self.trajectory_end_nodes if node.state == end_state]
        for i, current in enumerate(end_node):
            trajectory = [current.state]
            # rootノードまで一度辿る
            while True:
                parent_action = current.parent_action
                # 一つ上のノードに上がる
                current = current.parent_node
                # 根ノードであったなら終了
                if current == None:
                    break
                trajectory.append((current.state, parent_action))
            trajectory.reverse()
            print(i,end='')
            for s in trajectory:
                print(s,end='→')
            print()
    
    def trajectory_start_action(self, end_states):
        actions_count = [0 for i in range(len(self.env.action_space))]
        for end_state in end_states:
            end_node = [node for node in self.trajectory_end_nodes if node.state == end_state]
            # 各ac
            for current in end_node:
                # rootノードまで一度辿る
                while True:
                    parent_action = current.parent_action
                    # 一つ上のノードに上がる
                    current = current.parent_node
                    # 2つ上が根ノードなら終了(選択されたactionは子ノードが持っているため)
                    if current.parent_node == None:
                        actions_count[parent_action] += 1
                        break
        return actions_count

    def calculate_Emotion(self, pi, gamma):
        for end_node in self.trajectory_end_nodes:
            # end_node - 1から利用する
            node = end_node
            state = node.state
            b = []
            reward = []

            while node.parent_node != None:
                next_state = state
                # state時に取ったaction
                action = node.parent_action
                node = node.parent_node
                state = node.state
                p = self.env.get_prob_trans(state, action)
                b.append(pi[state][action] * p[next_state])
                reward.append(math.pow(gamma, node.depth)*self.env.reward(state,action,next_state))
                #if self.env.check_done(end_node.state):
                #    print(state, next_state, self.env.reward(state, action, next_state))
            
            belief = 1
            for item in b:
                belief *= item
            Vend = self.V(end_node.state, pi)
            Vs0 = self.V(node.state, pi)
            desire = sum(reward) + math.pow(gamma, end_node.depth+1)* Vend - Vs0

            emotion_value = belief * desire
            self.fears[end_node.state] += emotion_value if emotion_value < 0 else 0
            self.hopes[end_node.state] += emotion_value if emotion_value > 0 else 0



    def fear(self):
        value = min(list(self.fears.values()))
        state = [k for k, v in self.fears.items() if v == value]
        return state, value
    
    def hope(self):
        value = max(list(self.hopes.values()))
        state = [k for k, v in self.hopes.items() if v == value]
        return state, value

    #def V(self, state, pi):
    #    values = []
    #    for action in self.get_action_space(state):
    #        values.append(pi[state][action]*self.Q[state,action])
    #    return sum(values)

    def V(self, state, pi):
        qs = [self.Q[state, a] for a in self.get_action_space(state)]
        return max(qs)


if __name__ == '__main__':
    env = GridWorld()
    Q = defaultdict(lambda: 0)
    mcts = MCTS_T(env, Q, env.start_state, max_depth=5, loop=2000)

    random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    pi = defaultdict(lambda: random_actions)
    gamma = 1.0
    mcts.calculate_Emotion(pi, gamma)
    a, b = mcts.fear()
    print(a, b)
    a, b = mcts.hope()
    print(a, b)
    