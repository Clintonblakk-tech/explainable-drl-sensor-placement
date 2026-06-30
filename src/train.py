import argparse
import numpy as np
import torch
from environment import WaterDistributionEnv
from dqn_agent import DQNAgent

def train_pipeline(network_name, total_episodes, batch_size=32):
    print(f"Initializing optimization training loop on network benchmark: {network_name.upper()}")
    
    # Initialize environment wrapper and map dimensional boundaries
    env = WaterDistributionEnv(network_name=network_name, max_sensors=5)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Instantiate action-masked DQN optimization agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    for episode in range(1, total_episodes + 1):
        state, action_mask = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Action selection enforced by algorithmic safety action masking
            action = agent.act(state, action_mask)
            
            # Step inside hydraulic simulation boundary
            next_state, reward, done, next_mask = env.step(action)
            
            # Record trajectory inside experience replay buffer memory
            agent.remember(state, action, reward, next_state, float(done), next_mask)
            
            state = next_state
            action_mask = next_mask
            episode_reward += reward
            
            # Execute background training optimization step
            agent.replay(batch_size)
            
        # Periodically update stable target network parameters
        if episode % 10 == 0:
            agent.update_target_network()
            
        if episode % 50 == 0 or episode == 1:
            print(f"Episode: {episode}/{total_episodes} | Cumulative Reward: {episode_reward:.2f} | Exploration (Epsilon): {agent.epsilon:.3f}")
            
    print(f"Optimized layout training complete. Saving policy weights to model checkpoints...")
    # Save trained network state dictionary weights for downstream post-hoc evaluation and SHAP analysis
    torch.save(agent.model.state_dict(), f"src/dqn_{network_name}_policy.pt")
    print("Policy weights saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-DRL Optimization Engine for Sensor Placement Optimization.")
    parser.add_argument("--network", type=str, default="net3", help="Target benchmark network configuration (hanoi, net3, ctown).")
    parser.add_argument("--episodes", type=str, default="500", help="Total operational episodes for policy optimization.")
    
    args = parser.parse_args()
    
    try:
        ep_count = int(args.episodes)
    except ValueError:
        ep_count = 500
        
    train_pipeline(network_name=args.network, total_episodes=ep_count)
