from pacmanlearn import State
import random
from math import log
import copy
import config
import time

class MCTS_node:
    def __init__(self, state: State):
        self.state = state
        self.n = 0
        self.T = 0
        self.children = []
        self.parent = None
        self.alive = True #false for game end state

    def is_leaf(self):
        return len(self.children)==0
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def expand(self):
        if self.alive == False:
            return False
        possible_moves = self.state.valid_moves()
        for move in possible_moves:
            new_state = copy.deepcopy(self.state)
            result = new_state.move(move)
            child = MCTS_node(new_state)
            child.alive = result
            self.add_child(child)
        return True
    
    def is_alive(self):
        return len(self.children)>0 and self.alive
    
    def get_score(self):
        return self.T/self.n if self.n != 0 else 0
    
    def __str__(self, level=0):
        # 递归打印节点信息
        ret = "\t" * level + f"Node: n={self.n}, T={self.T}, score={self.get_score():.4f}, alive={self.alive}\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

class MCTS:
    def __init__(self, init_state:State, c=config.mcts_c):
        self.current_player = 'E'
        self.c = c
        self.root = MCTS_node(init_state)
        self.max_rollout = config.max_rollout

    def print_self(self):
        print(str(self.root))

    def UCB(self, node: MCTS_node):
        if node.n == 0:
            return float('inf')
        return node.T/node.n + self.c*(log(self.root.n)/node.n)**0.5
    
    def select(self):
        current_node = self.root
        max_point = -1
        while True:
            if current_node.is_leaf():
                return current_node
            current_select = current_node.children[0]
            for child in current_node.children:
                ucb = self.UCB(child)
                if ucb > max_point:
                    max_point = ucb
                    current_select = child
            current_node = current_select

    def rollout(state: State, left: int):# remember to copy state before rollout()
        if left == 0:
            return state.points #rollout max times reached
        left -= 1
        valid_moves = state.valid_moves()
        move = random.choice(valid_moves)
        if(state.move(move)):
            return MCTS.rollout(state, left)
        else:
            return state.points #got caught by guard
        
    def bp(node: MCTS_node, value: float):
        node.T += value
        node.n += 1
        if node.parent:
            MCTS.bp(node.parent, value)
    
    def run(self, max_runtime = config.max_runtime):
        start_time = time.time()
        while True:
            leafnode = self.select() #select best child recursively until reaches leaf node
            leafnode.expand() #expansion
            if leafnode.is_alive():
                cp_state = copy.deepcopy(leafnode.children[0].state)
                value = MCTS.rollout(cp_state, self.max_rollout)
                MCTS.bp(leafnode.children[0], value)
            else:
                MCTS.bp(leafnode, leafnode.state.points)
            if time.time() - start_time > max_runtime:
                break

    def decide(self):
        possible_moves = self.root.state.valid_moves()
        best_score = 0
        best_index = -1
        for i, child in enumerate(self.root.children):
            score = child.get_score()
            if score > best_score:
                best_score = score
                best_index = i 
        return possible_moves[best_index]