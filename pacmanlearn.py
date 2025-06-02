import copy
import config

class State:
    def __init__(self, map):
        self.map = copy.deepcopy(map)
        self.pos = {
            'E': (-1, -1),
            'G': (-1, -1)
        }
        self.points = 0
        self.movesleft = config.max_moves #距离游戏结束还剩多少步
        self.current_player = 'E'

        for i, row in enumerate(map):
            for j, col in enumerate(row):
                if col == 'E':
                    self.pos['E'] = (i, j)
                elif col == 'G':
                    self.pos['G'] = (i, j)

    def update_position(pos, direction):
        x, y = pos
        if direction == 'W':
            x -= 1
        elif direction == 'S':
            x += 1
        elif direction == 'A':
            y -= 1
        elif direction == 'D':
            y += 1
        elif direction == 'E':
            pass
        return x, y
    
    def valid_moves(self, player=None):
        moves = ['W', 'S', 'A', 'D', 'E']
        valid_moves = []
        if player is None:
            player = self.current_player
        for move in moves:
            newx, newy = State.update_position(self.pos[player], move)
            if self.map[newx][newy] != 'X':
                valid_moves.append(move)
        return valid_moves
    
    def switch_player(self):
        self.current_player = 'E' if self.current_player=='G' else 'G'
    
    def move(self, direction, player=None):
        assert direction in ['E', 'W', 'A', 'S', 'D']
        if player is None:
            player = self.current_player
        self.movesleft -= 1
        newx, newy = State.update_position(self.pos[player], direction)
        if self.map[newx][newy] == ' ':
            self.pos[player] = newx, newy
        elif self.map[newx][newy] == '.':
            self.pos[player] = newx, newy
            self.points += 1
        elif self.map[newx][newy] == 'X':
            pass
        else:
            #meets another agent
            return False
        if self.movesleft <= 0:
            return False
        self.switch_player()
        return True
