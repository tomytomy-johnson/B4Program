from collections import defaultdict

class Node:
    def __init__(self, state, depth, parent_node, parent_action):
        # 状態
        self.state = state
        # 選択回数(通った回数)
        self.count_num = 0
        # 深さ
        self.depth = depth
        # 親Nodeと選択されたaction
        self.parent_node = parent_node
        self.parent_action = parent_action

        # 遷移先ノード{(action, next_state): next_Node(next_state,)}
        self.next_nodes = defaultdict(lambda: None)

        # 各行動に対する不確かさ(探索強度) sigma -> [0,1]
        self.sigma = defaultdict(lambda:1.0)

        #act_name = self.parent_action if self.parent_action != None else None
        #parent_node = parent_node.state if self.parent_node != None else None
        #print("depth: "+str(self.depth)+"| Node "+str(parent_node)+" ,Action("+str(act_name)+") → Node [" + str(self.state) + "]")
