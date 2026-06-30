import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = decay
        
        self.memory = deque(maxlen=20000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DeepQNetwork(state_dim, action_dim).to(self.device)
        self.target_model = DeepQNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, action_mask):
        self.memory.append((state, action, reward, next_state, done, action_mask))

    def act(self, state, action_mask):
        if np.random.rand() <= self.epsilon:
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t).cpu().data.numpy()[0]
        
        masked_q_values = np.where(action_mask == 1, q_values, -1e9)
        return np.argmax(masked_q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*minibatch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states_t).cpu().numpy()
            masked_next_q = np.where(np.array(next_masks) == 1, next_q_values, -1e9)
            max_next_q = torch.FloatTensor(np.max(masked_next_q, axis=1)).to(self.device)
        
        target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
