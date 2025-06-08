from pacmanlearn import State
import config
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def lr_lambda(epoch):
    return max(config.DQNlrDecay ** epoch, config.DQNMinlr / config.DQNlr)

class DQN:
    def __init__(self, fisrt_state: State):
        self.first_state = copy.deepcopy(fisrt_state)
        self.steppunish = config.DQNsteppunish
        self.q_network = DQNNetwork()
        #target network has the same structure and init parameters as q_network
        self.target_network = DQNNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.epsilon = config.DQNepsilon  # Exploration rate
        self.gamma = config.DQNgamma  # Discount factor
        self.optimizer1 = torch.optim.Adam(self.q_network.parameters(), lr=config.DQNlr)
        self.scheduler1 = torch.optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda)
        self.optimizer2 = torch.optim.Adam(self.target_network.parameters(), lr=config.DQNlr)
        self.scheduler2 = torch.optim.lr_scheduler.LambdaLR(self.optimizer2, lr_lambda)
        self.batch_size = config.DQNbatch_size
        self.replay_buffer = []  # Experience replay buffer
        self.epochs = config.DQNepochs  # Number of training epochs

    def reward(self, state: State, action: str):
        cp_state = copy.deepcopy(state)
        before = cp_state.points
        alive = cp_state.move(action)
        if ((not alive) and (cp_state.movesleft>0)) or cp_state.dead_man_walking():
            return -100
        after = cp_state.points
        return after - before - self.steppunish

    def encode(state: State):
        # 将状态编码为一个向量，默认编码的状态是E的回合
        map_data = np.zeros((4, len(state.map), len(state.map[0])), dtype=np.float32)
        # 四个通道分别为豆子、墙壁、E位置和G位置
        for i in range(len(state.map)):
            for j in range(len(state.map[0])):
                if state.map[i][j] == '.':
                    map_data[0][i][j] = 1.0
                elif state.map[i][j] == 'X':
                    map_data[1][i][j] = 1.0
                elif state.map[i][j] == 'E':
                    map_data[2][i][j] = 1.0
                elif state.map[i][j] == 'G':
                    map_data[3][i][j] = 1.0
        return torch.tensor(map_data, dtype=torch.float32)    
    
    def decode(encoded_state: torch.Tensor):
        # 将编码的状态解码为一个State对象
        map_data = np.zeros((len(encoded_state[0]), len(encoded_state[0][0])), dtype=str)
        for i in range(len(encoded_state[0])):
            for j in range(len(encoded_state[0][0])):
                if encoded_state[0][i][j] == 1.0:
                    map_data[i][j] = '.'
                elif encoded_state[1][i][j] == 1.0:
                    map_data[i][j] = 'X'
                elif encoded_state[2][i][j] == 1.0:
                    map_data[i][j] = 'E'
                elif encoded_state[3][i][j] == 1.0:
                    map_data[i][j] = 'G'
                else:
                    map_data[i][j] = ' '
        return map_data

    def train(self):
        losses = []

        current_state = copy.deepcopy(self.first_state)
        dead = False
        for epoch in range(self.epochs):
            if dead:
                current_state = copy.deepcopy(self.first_state)
                dead = False
            encode_s = DQN.encode(current_state)
            # 1. Epsilon-greedy action selection
            action = None
            q_values = None
            if np.random.rand() < self.epsilon:
                action = np.random.choice(['W', 'S', 'A', 'D', 'E'])
            else:
                with torch.no_grad():
                    self.q_network.eval()
                    encoded_state = encode_s.unsqueeze(0)  # Add batch dimension
                    q_values = self.q_network(encoded_state)
                    action_index = torch.argmax(q_values).item()  # Get the index of the max Q-value
                    action = ['W', 'S', 'A', 'D', 'E'][action_index]  # Convert index to action
            # 2. Execute action and observe next state and reward
            reward = self.reward(current_state, action)
            # 3. Store experience in replay buffer
            dead = not current_state.move(action)

            # Extra step: guard turn
            # Because we only train the E agent, we need to simulate the guard's turn
            current_state.switch_player()
            guard_moves = current_state.valid_moves('G')
            # epsilon-greedily select the move that minimizes Q-value for the E agent
            if guard_moves:
                if np.random.rand() < self.epsilon:
                    guard_action = np.random.choice(guard_moves)
                    current_state.move(guard_action, 'G')
                else:
                    with torch.no_grad():
                        self.target_network.eval()
                        guard_q_values = self.target_network(DQN.encode(current_state).unsqueeze(0))
                        guard_action_index = torch.argmin(guard_q_values).item()
                        guard_action = ['W', 'S', 'A', 'D', 'E'][guard_action_index]
                        current_state.move(guard_action, 'G')
            # Switch back to E's turn
            current_state.switch_player()

            self.replay_buffer.append((encode_s, action, reward, DQN.encode(current_state)))
            if len(self.replay_buffer) > config.DQNMaxBufferSize:
                self.replay_buffer.pop(0)
            # 4. Sample a batch from the replay buffer
            if len(self.replay_buffer) < self.batch_size:
                continue
            batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            states, actions, rewards, next_states = zip(*[self.replay_buffer[i] for i in batch])
            states = torch.stack(states)
            actions = torch.tensor([['W', 'S', 'A', 'D', 'E'].index(a) for a in actions], dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.stack(next_states)
            # 5. Compute target Q-values
            with torch.no_grad():
                self.target_network.eval()
                next_q_values = self.target_network(next_states) #KEY IDEA! USE TARGET NETWORK TO COMPUTE NEXT Q VALUES
                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                target_q_values = rewards + self.gamma * max_next_q_values
            # 6. Gradient descent step
            self.q_network.train()
            self.optimizer1.zero_grad()
            q_values = self.q_network(states)
            # Gather the Q-values for the actions taken
            action_indices = actions.unsqueeze(1)  # Reshape to (batch_size, 1)
            q_values_for_actions = q_values.gather(1, action_indices).squeeze(1)  # Shape: (batch_size,)
            # Compute the loss
            loss = nn.MSELoss()(q_values_for_actions, target_q_values)
            # print("q_values_for_actions:", q_values_for_actions.shape)
            # print("target_q_values:", target_q_values.shape)
            # print("loss:", loss.item())
            loss.backward()
            self.optimizer1.step()
            self.scheduler1.step()  # Update learning rate
            # 7. Update target network
            if epoch % config.DQNUpdateFreq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            # 8. Print progress
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
                losses.append(loss.item())
            # 9. Decay epsilon
            self.epsilon = max(config.DQNMinEpsilon, self.epsilon * config.DQNEpsilonDecay)  # Decay epsilon
            # 10. Save model periodically
            if epoch % config.DQNSaveFreq == 0:
                torch.save(self.q_network.state_dict(), f'models/dqn_model_epoch_{epoch}.pth')
                print(f"Model saved at epoch {epoch} as 'dqn_model_epoch_{epoch}.pth'.")

        # save the trained model
        torch.save(self.q_network.state_dict(), 'models/dqn_model.pth')
        print("Training complete. Model saved as 'dqn_model.pth'.")
        # save the replay buffer as txt file
        with open('models/dqn_replay_buffer.txt', 'w') as f:
            for state, action, reward, next_state in self.replay_buffer:
                f.write(f"{DQN.decode(state.tolist())}\n{action}\n{reward}\n{DQN.decode(next_state.tolist())}\n\n")
        print("Replay buffer saved as 'dqn_replay_buffer.txt'.")

        # Plot the loss curve
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('DQN Training Loss')
        plt.show()

class DQNNetwork(nn.Module):
    def __init__(self, map_size = (12, 7), map_channels = 4, map_action = 5, fc_hidden_size = config.FCHiddenSize):
        super(DQNNetwork, self).__init__()
        self.map_size = map_size
        self.map_action = map_action

        # Define the neural network architecture
        # Input size is batchsize * 4 * x * y (for the four channels)
        # Use CNN to process the map data
        self.cnn1 = nn.Conv2d(map_channels, 16, kernel_size=5, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # Now the output size is batchsize * 16 * (x-4) * (y-4)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Now the output size is batchsize * 32 * (x-6) * (y-6)
        self.cnn3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=2)
        self.relu3 = nn.ReLU()
        # Flatten the output for the fully connected layers
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc1 = nn.Linear(4* 32 * (map_size[0] - 6) * (map_size[1] - 6), fc_hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.relu6 = nn.ReLU()
        self.fc4 = nn.Linear(fc_hidden_size, map_action)

    def forward(self, x):
        # x is expected to be of shape (batch_size, 4, height, width)
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.cnn3(x)
        x = self.relu3(x)
        x = self.flatten(x)
    
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.relu6(x)
        x = self.fc4(x)
        return x
    
    def decide(self, state: State):
        # 选择动作
        encode_s = DQN.encode(state)
        with torch.no_grad():
            self.eval()
            encoded_state = encode_s.unsqueeze(0)
            q_values = self.forward(encoded_state)
            action_index = torch.argmax(q_values).item()
            action = ['W', 'S', 'A', 'D', 'E'][action_index]  # Convert index to action
        return action
    
if __name__ == "__main__":
    map_data = []
    # read map from map.txt
    with open('map.txt', 'r') as file:
        for line in file:
            row = list(line.strip())
            map_data.append(row)
    map_state = State(map_data)
    dqn_agent = DQN(map_state)
    dqn_agent.train()
