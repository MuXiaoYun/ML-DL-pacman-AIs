import os
import sys
import time
import threading
from queue import Queue
import config
from pacmanlearn import State
from mcts import MCTS
from dqn import DQNNetwork
import torch

# 全局变量
map_data = []
pacman_pos = None
guard_pos = None
score = 0
input_queue = Queue()
ai = {
        1: "E",
        2: "G"
    }
last_ai_move = 'NONE'
moves_left = config.max_moves

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_map(file_path):
    global map_data, pacman_pos, guard_pos
    with open(file_path, 'r') as file:
        for line in file:
            row = list(line.strip())
            map_data.append(row)
            if 'E' in row:
                pacman_pos = (len(map_data) - 1, row.index('E'))
            if 'G' in row:
                guard_pos = (len(map_data) - 1, row.index('G'))

def print_map(current_turn = "None"):
    global last_ai_move, moves_left
    clear_screen()
    print(f"Score: {score}")
    print(f"Moves left: {moves_left}")
    print(f"Current: {current_turn}")
    print(f"LastAiMove: {last_ai_move}")
    for row in map_data:
        print(''.join(f"{cell:^3}" for cell in row))

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

def handle_input():
    global input_queue
    while True:
        input_char = sys.stdin.read(1)
        input_queue.put(input_char)

def end_game():
    clear_screen()
    print("GAME OVER!")
    print(f"SCORE: {score}")

def move_character(character, direction):
    global map_data, pacman_pos, guard_pos, score, moves_left
    moves_left -= 1
    if character == 'E':
        current_pos = pacman_pos
        new_pos = update_position(current_pos, direction)
        if 0 <= new_pos[0] < len(map_data) and 0 <= new_pos[1] < len(map_data[0]):
            if map_data[new_pos[0]][new_pos[1]] != 'X':
                if map_data[new_pos[0]][new_pos[1]] == '.':
                    score += 1
                if map_data[new_pos[0]][new_pos[1]] == 'G':
                    end_game()
                    return True
                map_data[current_pos[0]][current_pos[1]] = ' '
                map_data[new_pos[0]][new_pos[1]] = 'E'
                pacman_pos = new_pos
    elif character == 'G':
        current_pos = guard_pos
        new_pos = update_position(current_pos, direction)
        if 0 <= new_pos[0] < len(map_data) and 0 <= new_pos[1] < len(map_data[0]):
            if map_data[new_pos[0]][new_pos[1]] != 'X':
                if map_data[new_pos[0]][new_pos[1]] == 'E':
                    end_game()
                    return True
                map_data[current_pos[0]][current_pos[1]] = ' '
                map_data[new_pos[0]][new_pos[1]] = 'G'
                guard_pos = new_pos
    return False

def ai_controling(current_turn):
    global ai
    if ai[config.ai_controls] == current_turn:
        return True
    return False

def main():
    global input_queue, ai, last_ai_move, moves_left

    if config.ai_type == 'dqn':
        dqn = DQNNetwork()           
        dqn.load_state_dict(torch.load('models/dqn_model.pth'))

    load_map('map.txt')
    input_thread = threading.Thread(target=handle_input)
    input_thread.daemon = True
    input_thread.start()
    current_turn = 'E'
    print_map(current_turn)
    while True:
        if ai_controling(current_turn):
            ## ai here
            try:
                if config.ai_type == 'mcts':
                    mcts = MCTS(State(map_data))
                    mcts.run()
                    move = mcts.decide()
                    last_ai_move = move
                    if move_character(current_turn, move):
                        break
                    current_turn = 'G' if current_turn == 'E' else 'G'
                    print_map(current_turn)
                    time.sleep(0.1)
                elif config.ai_type == 'dqn':
                    move = dqn.decide(State(map_data))
                    last_ai_move = move
                    if move_character(current_turn, move):
                        break
                    current_turn = 'G' if current_turn == 'E' else 'E'
                    print_map(current_turn)
                    time.sleep(0.1)
                else:
                    raise NotImplementedError("AI type not implemented")
            except Exception as e:
                print("----------------ERROR!----------------")
                print(f"Exception: {e}")
                if config.ai_type == 'mcts':
                    mcts.print_self()
                raise    
            continue
        if not input_queue.empty():       
            input_char = input_queue.get()
            if current_turn == 'E' and input_char.upper() in ['W', 'A', 'S', 'D', 'E']:
                if move_character('E', input_char.upper()):
                    break
                current_turn = 'G'
            elif current_turn == 'G' and input_char in ['I', 'J', 'K', 'L', 'O']:
                direction = {'I': 'W', 'K': 'S', 'L': 'D', 'J': 'A', 'O': 'E'}.get(input_char.upper())
                if move_character('G', direction):
                    break
                current_turn = 'E'
            print_map(current_turn)
            time.sleep(0.1)
        if moves_left == 0:
            break
    if config.ai_type == 'mcts':
        mcts.print_self()

if __name__ == "__main__":
    main()