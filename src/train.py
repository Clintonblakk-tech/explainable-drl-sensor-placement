"""Run one new DQN training experiment. New runs are not historical results."""
import argparse,csv,json,random,sys,time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(Path(__file__).parent))
from environment import SensorPlacementEnvironment
from dqn_agent import DQNAgent
FILES={"Hanoi":ROOT/"networks"/"Hanoi.inp","Net3":ROOT/"networks"/"Net3.inp","C-Town":ROOT/"networks"/"C-Town.inp"}; BUDGETS={"Hanoi":3,"Net3":5,"C-Town":10}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--network',choices=FILES,required=True); p.add_argument('--episodes',type=int,default=1000); p.add_argument('--seed',type=int,default=1); p.add_argument('--budget',type=int); p.add_argument('--outdir',type=Path,default=ROOT/'results'/'new_runs'); a=p.parse_args(); budget=a.budget or BUDGETS[a.network]
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); env=SensorPlacementEnvironment(FILES[a.network],budget); agent=DQNAgent(len(env.nodes),len(env.nodes)); rows=[]; best=-float('inf'); start=time.perf_counter(); last_info={}
 for ep in range(a.episodes):
  state=env.reset(); done=False; total=0.0
  while not done:
   action=agent.act(state,env.valid_actions()); ns,r,done,info=env.step(action); agent.remember((state.copy(),action,r,ns.copy(),done)); agent.update(); state=ns; total+=r; last_info=info
  best=max(best,total); rows.append({'episode':ep+1,'episode_reward':total,'epsilon':agent.epsilon}); agent.finish_episode(ep)
 elapsed=time.perf_counter()-start; a.outdir.mkdir(parents=True,exist_ok=True); stem=f"{a.network.replace('-','_')}_budget{budget}_seed{a.seed}"
 with (a.outdir/f'{stem}_episode_history.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 torch.save(agent.online.state_dict(),a.outdir/f'{stem}_qnetwork.pth'); summary={'network':a.network,'budget':budget,'seed':a.seed,'episodes':a.episodes,'best_episode_return':best,'final_episode_return':rows[-1]['episode_reward'],'runtime_seconds':elapsed,'final_episode_nodes':last_info.get('selected_nodes',[]),'status':'new run; not a historical run reconstruction'}; (a.outdir/f'{stem}_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
