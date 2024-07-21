from operator import truediv, add
from turtle import done
import numpy as np
import common.gridworld_render_CliffWalking as render_helper


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array(
            [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
             [-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1]]
        )
        self.goal_state = (3, 11)
        self.start_state = (3, 0)
        # Cliff
        self.cliff_states = [
            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]

        # self.reward_map = np.array(
        #    [[-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
        #     [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
        #     [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
        #     [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
        #     [-0.001, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.02]]
        # )
        #self.goal_state = (4, 11)
        #self.start_state = (4, 0)
        # Cliff
        # self.cliff_states = [
        #    (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10)]

        self.wall_state = (None, None)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    # 遷移確率をdictで返す
    def get_prob_trans(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 移動先が壁なら移動せずそのまま(ターンは消費する)
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        # 移動先が崖の隣(y軸方向1つ上)ならslip_prob=10%の確率で崖に落ちる
        #for cliff in self.cliff_states:
        #   # 移動先が崖なら関係ない(100%)
        #   if next_state == cliff:
        #       break
        #   slipper = (cliff[0] - 1, cliff[1])
        #   if state == slipper:
        #       slip_prob = 0.05
        #       return {next_state: 1.0-slip_prob, cliff: slip_prob}

        return {next_state: 1.0}

    # 遷移する可能性のある状態リストを返す
    def get_trans_states(self, state, action):
        d_states = (self.get_prob_trans(state, action)).keys()
        return list(d_states)

    # 遷移確率に従って次状態に遷移する
    def next_state(self, state, action):
        next_state_probs = self.get_prob_trans(state, action)
        next_states = list(next_state_probs.keys())
        # random.choiceを使うために番号を割り当てる
        map_states = {k: v for k, v in zip(
            range(len(next_states)), next_states)}
        probs = list(next_state_probs.values())
        key = np.random.choice(list(map_states.keys()), p=probs)
        return map_states[key]

    # def next_state(self, state, action):
    #    action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #    move = action_move_map[action]
    #    next_state = (state[0] + move[0], state[1] + move[1])
    #    ny, nx = next_state
    #    # 移動先が壁なら移動せずそのまま(ターンは消費する)
    #    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
    #        next_state = state
    #    elif next_state == self.wall_state:
    #        next_state = state
    #    return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = self.check_done(next_state)

        self.agent_state = next_state
        return next_state, reward, done

    def check_done(self, state):
        done = (state == self.goal_state)
        # Cliffでもdone
        for cliff in self.cliff_states:
            if state == cliff:
                done = True
                break
        return done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state, self.cliff_states)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state, self.cliff_states)
        renderer.render_q(q, print_value)

    def render_e(self, e=None):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state, self.cliff_states)
        renderer.render_e(e)

    def render_end_state(self, end_states=None):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state, self.cliff_states)
        renderer.render_end_state(end_states)
