"""DQN network and experience-learning utilities."""

from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Feed-forward Q-network for graph-based sensor placement."""

    def __init__(self, state_dim, action_dim):
        super().__init__()

        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        """Return Q-values for the supplied state or batch of states."""
        return self.net(x)


class DQNAgent:
    """DQN agent with optional experience replay and target network."""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0005,
        gamma=0.99,
        replay_capacity=50000,
        batch_size=64,
        use_replay=True,
        use_target=True,
    ):
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError(
                "state_dim and action_dim must both be positive."
            )

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1.")

        if replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.online = QNetwork(
            self.state_dim, self.action_dim
        ).to(self.device)

        self.target = QNetwork(
            self.state_dim, self.action_dim
        ).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(
            self.online.parameters(),
            lr=lr,
        )

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.use_replay = bool(use_replay)
        self.use_target = bool(use_target)

        self.memory = deque(maxlen=int(replay_capacity))
        self.last_transition = None

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state, valid):
        """Choose an available action using epsilon-greedy selection."""

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        valid = list(map(int, valid))

        if state.size != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}; "
                f"received {state.size}."
            )

        if not valid:
            raise ValueError("No valid actions are available.")

        if len(set(valid)) != len(valid):
            raise ValueError("The valid-action list contains duplicates.")

        if any(a < 0 or a >= self.action_dim for a in valid):
            raise ValueError("A valid action is outside the action space.")

        if random.random() < self.epsilon:
            return random.choice(valid)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            q_values = self.online(state_tensor)

            valid_tensor = torch.as_tensor(
                valid,
                dtype=torch.long,
                device=self.device,
            )

            valid_q_values = q_values[valid_tensor]
            best_index = int(valid_q_values.argmax().item())

            return valid[best_index]

    def remember(self, state, action, reward, next_state, done):
        """Store a transition and retain the latest transition."""

        state = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        next_state = np.asarray(
            next_state, dtype=np.float32
        ).reshape(-1).copy()

        action = int(action)
        reward = float(reward)
        done = bool(done)

        if state.size != self.state_dim:
            raise ValueError("Invalid state dimension in transition.")

        if next_state.size != self.state_dim:
            raise ValueError("Invalid next-state dimension in transition.")

        if not 0 <= action < self.action_dim:
            raise ValueError("Transition action is outside the action space.")

        if not np.all(np.isfinite(state)):
            raise ValueError("State contains non-finite values.")

        if not np.all(np.isfinite(next_state)):
            raise ValueError("Next state contains non-finite values.")

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite.")

        transition = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        self.memory.append(transition)
        self.last_transition = transition

    def update(self):
        """Perform one Q-learning update when sufficient data are available."""

        if self.use_replay:
            if len(self.memory) < self.batch_size:
                return None

            batch = random.sample(
                self.memory,
                self.batch_size,
            )

        else:
            # Genuine no-replay update: use only the latest transition.
            if self.last_transition is None:
                return None

            batch = [self.last_transition]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(
            np.stack(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.stack(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        predicted_q = self.online(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_network = (
                self.target if self.use_target else self.online
            )

            next_q_values = next_network(next_states)

            # Binary state entries identify already-selected sensors.
            # These locations are unavailable as next actions.
            next_q_values = next_q_values.masked_fill(
                next_states > 0.5,
                -torch.inf,
            )

            max_next_q = next_q_values.max(dim=1).values

            # Avoid invalid values if every action is masked.
            max_next_q = torch.where(
                torch.isfinite(max_next_q),
                max_next_q,
                torch.zeros_like(max_next_q),
            )

            target_q = (
                rewards
                + self.gamma * (1.0 - dones) * max_next_q
            )

        loss = nn.functional.mse_loss(
            predicted_q,
            target_q,
        )

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return float(loss.detach().cpu().item())

    def finish_episode(self, episode):
        """Decay exploration and synchronize the target network."""

        if episode < 0:
            raise ValueError("episode must be non-negative.")

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        if self.use_target and (episode + 1) % 100 == 0:
            self.target.load_state_dict(
                self.online.state_dict()
            )
