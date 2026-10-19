"""Graph-based placement environment; documented implementation of manuscript method."""
from pathlib import Path
import networkx as nx
import numpy as np
import wntr

class SensorPlacementEnvironment:
    def __init__(self, network_file, sensor_budget, coverage_weight=0.70, centrality_weight=0.30, coverage_radius=2):
        if not Path(network_file).exists(): raise FileNotFoundError(f"Missing benchmark INP: {network_file}")
        self.wn=wntr.network.WaterNetworkModel(str(network_file)); self.graph=self.wn.get_graph(); self.nodes=list(self.graph.nodes())
        if not self.nodes: raise ValueError("No graph nodes available")
        if not 1 <= sensor_budget <= len(self.nodes): raise ValueError("Invalid sensor budget")
        self.sensor_budget=int(sensor_budget); self.coverage_weight=float(coverage_weight); self.centrality_weight=float(centrality_weight); self.coverage_radius=int(coverage_radius)
        self.centrality=nx.degree_centrality(self.graph); self.reset()
    def reset(self): self.selected_nodes=[]; self.state=np.zeros(len(self.nodes),dtype=np.float32); return self.state.copy()
    def valid_actions(self): return np.flatnonzero(self.state==0)
    def coverage(self):
        covered=set()
        for n in self.selected_nodes: covered.update(nx.single_source_shortest_path_length(self.graph,n,cutoff=self.coverage_radius))
        return len(covered)/len(self.nodes)
    def step(self,action):
        action=int(action)
        if action<0 or action>=len(self.nodes) or self.state[action]==1: raise ValueError("Invalid/repeated action")
        self.selected_nodes.append(self.nodes[action]); self.state[action]=1.0
        cov=self.coverage(); cent=float(np.mean([self.centrality.get(n,0.0) for n in self.selected_nodes]))
        reward=self.coverage_weight*cov+self.centrality_weight*cent; done=len(self.selected_nodes)>=self.sensor_budget
        return self.state.copy(),float(reward),done,{"selected_nodes":[str(n) for n in self.selected_nodes],"coverage":cov,"mean_degree_centrality":cent}
