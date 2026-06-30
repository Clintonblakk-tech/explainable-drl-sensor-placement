import numpy as np

class WaterDistributionEnv:
    """
    Custom environment wrapper for benchmark water distribution networks.
    Simulates operational dynamics, tracks metrics, and enforces algorithmic action-masking.
    """
    def __init__(self, network_name="net3", max_sensors=5):
        self.network_name = network_name
        self.max_sensors = max_sensors
        
        # Simulating architectural layouts for demo benchmarking purposes
        # In full run execution, these dimensions align with total layout nodes
        if network_name.lower() == "hanoi":
            self.num_nodes = 32
        elif network_name.lower() == "ctown":
            self.num_nodes = 388
        else: # Defaulting to net3
            self.num_nodes = 92
            
        self.state_dim = self.num_nodes * 2  # Example tracking features (e.g., Pressure + Placement Status)
        self.action_dim = self.num_nodes
        self.reset()

    def reset(self):
        self.current_step = 0
        self.placed_sensors = []
        
        # Action Mask representation: 1 = Available/Valid node placement, 0 = Illegal/Masked node placement
        self.action_mask = np.ones(self.action_dim, dtype=np.float32)
        
        # Standard structural initialization of state vectors
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        return self.state, self.action_mask

    def step(self, action):
        """Executes a placement step, calculates multi-objective rewards, and returns next state values."""
        if self.action_mask[action] == 0:
            raise ValueError(f"Algorithmic Safety Violation: Action {action} was masked but selected.")
            
        # Update architectural records and apply state transformation updates
        self.placed_sensors.append(action)
        self.action_mask[action] = 0.0  # Mask out this node to completely prevent duplicate selections
        
        # Reflect chosen layout in the state array vector flags
        self.state[action] = 1.0 
        # Add random synthetic variance to remaining feature slots to mimic operational water hydraulics
        self.state[self.num_nodes:] = np.random.normal(0.5, 0.15, self.num_nodes)
        
        self.current_step += 1
        done = len(self.placed_sensors) >= self.max_sensors
        
        # Formulate Multi-Objective Reward: balances leak isolations, target bounds, and configuration cost tracking
        reward = 1.5 if action % 3 == 0 else 0.2  # Toy model evaluation proxy logic
        if done:
            reward += 5.0  # Added terminal reward bonuses for reaching optimal layout target parameters
            
        return self.state, reward, done, self.action_mask
