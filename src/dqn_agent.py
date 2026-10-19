"""DQN network and replay utilities."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__(); self.net=nn.Sequential(nn.Linear(state_dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,action_dim))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self,state_dim,action_dim,lr=0.0005,gamma=0.99,replay_capacity=50000,batch_size=64,use_replay=True,use_target=True):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.online=QNetwork(state_dim,action_dim).to(self.device); self.target=QNetwork(state_dim,action_dim).to(self.device); self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt=optim.Adam(self.online.parameters(),lr=lr); self.gamma=gamma; self.batch_size=batch_size; self.use_replay=use_replay; self.use_target=use_target; self.memory=deque(maxlen=replay_capacity); self.epsilon=1.0; self.epsilon_min=0.01; self.epsilon_decay=0.995
    def act(self,state,valid):
        valid=list(map(int,valid))
        if not valid: raise ValueError("No valid actions")
        if random.random()<self.epsilon: return random.choice(valid)
        with torch.no_grad():
            q=self.online(torch.tensor(state,dtype=torch.float32,device=self.device)); mask=torch.full_like(q,-torch.inf); ids=torch.tensor(valid,dtype=torch.long,device=self.device); mask[ids]=q[ids]; return int(mask.argmax().item())
    def remember(self,*transition): self.memory.append(transition)
    def update(self):
        if len(self.memory)<(self.batch_size if self.use_replay else 1): return
        batch=random.sample(self.memory,self.batch_size if self.use_replay else 1); s,a,r,ns,d=zip(*batch)
        s=torch.tensor(np.asarray(s),dtype=torch.float32,device=self.device); a=torch.tensor(a,dtype=torch.long,device=self.device); r=torch.tensor(r,dtype=torch.float32,device=self.device); ns=torch.tensor(np.asarray(ns),dtype=torch.float32,device=self.device); d=torch.tensor(d,dtype=torch.float32,device=self.device)
        pred=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            q=(self.target if self.use_target else self.online)(ns); q=q.masked_fill(ns>0.5,-torch.inf); mx=q.max(1).values; mx=torch.where(torch.isfinite(mx),mx,torch.zeros_like(mx)); target=r+self.gamma*(1-d)*mx
        loss=nn.functional.mse_loss(pred,target); self.opt.zero_grad(); loss.backward(); self.opt.step()
    def finish_episode(self,episode):
        self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)
        if self.use_target and (episode+1)%100==0: self.target.load_state_dict(self.online.state_dict())
