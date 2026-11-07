"""Reproducible tabular Q-learning for a two-patch foraging MDP."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CAPACITY=3
HORIZON=40
ACTIONS=('stay','switch')
SEEDS=list(range(20))
TRAIN_EPISODES=2500
EVAL_EPISODES=200

class ForagingEnv:
    """State=(current_patch, food_A, food_B), 2*4*4=32 states."""
    def __init__(self, seed=0):
        self.rng=np.random.default_rng(seed)
        self.reset()
    def reset(self):
        self.patch=int(self.rng.integers(2))
        self.food=[int(self.rng.integers(1,4)),int(self.rng.integers(1,4))]
        self.t=0
        return self.state()
    def state(self): return (self.patch,self.food[0],self.food[1])
    def step(self, action):
        assert action in (0,1)
        travel=0.65 if action==1 else 0.0
        if action==1: self.patch=1-self.patch
        # consume one food unit if available; patch A richer than B
        harvest=(2.2 if self.patch==0 else 1.6) if self.food[self.patch]>0 else 0.0
        if self.food[self.patch]>0: self.food[self.patch]-=1
        # Each depleted slot regenerates independently with patch-specific probability.
        for p,regen_prob in enumerate((0.32,0.55)):
            if self.food[p]<CAPACITY and self.rng.random()<regen_prob:
                self.food[p]+=1
        self.t+=1
        return self.state(), harvest-travel, self.t>=HORIZON

def encode(state):
    p,a,b=state
    return p*16+a*4+b

def train(seed):
    env=ForagingEnv(seed)
    rng=np.random.default_rng(seed+100000)
    q=np.zeros((32,2),dtype=float)
    rewards=[]
    for ep in range(TRAIN_EPISODES):
        state=env.reset()
        epsilon=max(0.05,1.0-0.95*ep/1800)
        total=0.0
        done=False
        while not done:
            s=encode(state)
            action=int(rng.integers(2)) if rng.random()<epsilon else int(np.argmax(q[s]))
            ns,reward,done=env.step(action)
            target=reward+(0 if done else 0.95*np.max(q[encode(ns)]))
            q[s,action]+=0.12*(target-q[s,action])
            total+=reward
            state=ns
        rewards.append(total)
    return q,np.asarray(rewards)

def evaluate(seed, policy, q=None):
    env=ForagingEnv(seed)
    rng=np.random.default_rng(seed+200000)
    scores=[]
    for _ in range(EVAL_EPISODES):
        state=env.reset()
        done=False
        score=0
        while not done:
            if policy=='random': action=int(rng.integers(2))
            elif policy=='heuristic':
                p,a,b=state
                # Switch if current patch is empty and the other has food;
                # otherwise stay to avoid switching cost.
                food=(a,b)
                action=int(food[p]==0 and food[1-p]>0)
            elif policy=='qlearning': action=int(np.argmax(q[encode(state)]))
            else: raise ValueError(policy)
            state,r,done=env.step(action)
            score+=r
        scores.append(score)
    return np.array(scores)

def main():
    curves=[]
    results={key:[] for key in ('qlearning','heuristic','random')}
    for seed in SEEDS:
        q,curve=train(seed)
        curves.append(curve)
        for method in results:
            results[method].append(evaluate(50000+seed,method,q if method=='qlearning' else None).mean())
    curves=np.asarray(curves)
    means={k:float(np.mean(v)) for k,v in results.items()}
    stds={k:float(np.std(v,ddof=1)) for k,v in results.items()}
    summary={'environment':{'states':32,'actions':2,'horizon':HORIZON,'patch_A_harvest':2.2,'patch_B_harvest':1.6,'patch_A_regeneration_probability':0.32,'patch_B_regeneration_probability':0.55,'switch_cost':0.65},'training':{'independent_seeds':len(SEEDS),'episodes_per_seed':TRAIN_EPISODES,'evaluation_episodes_per_seed':EVAL_EPISODES,'alpha':0.12,'gamma':0.95,'epsilon_start':1.0,'epsilon_end':0.05},'mean_episode_reward':means,'seed_level_std':stds,'seed_level_rewards':results}
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    x=np.arange(1,TRAIN_EPISODES+1)
    window=100
    smooth=np.array([np.convolve(row,np.ones(window)/window,mode='valid') for row in curves])
    plt.figure(figsize=(9,4.8))
    xx=x[window-1:]
    plt.plot(xx,smooth.mean(axis=0),label='Q-learning (100-episode moving average)',linewidth=2)
    plt.fill_between(xx,smooth.mean(axis=0)-smooth.std(axis=0),smooth.mean(axis=0)+smooth.std(axis=0),alpha=.2,label='±1 SD across 20 seeds')
    plt.xlabel('Training episode');plt.ylabel('Total reward per episode');plt.title('Q-learning training curve');plt.legend();plt.grid(alpha=.2);plt.tight_layout();plt.savefig(OUT/'learning_curve.png',dpi=180);plt.close()
    plt.figure(figsize=(8,4.8))
    keys=['random','heuristic','qlearning']
    bars=plt.bar(['Random','Heuristic','Q-learning'],[means[k] for k in keys],yerr=[stds[k] for k in keys],capsize=6)
    plt.ylabel('Mean reward per 40-step episode');plt.title('Evaluation: 20 independent seeds × 200 episodes');plt.grid(axis='y',alpha=.2)
    for b,k in zip(bars,keys): plt.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{means[k]:.2f}',ha='center',fontsize=10)
    plt.tight_layout();plt.savefig(OUT/'baseline_comparison.png',dpi=180);plt.close()
    print(json.dumps({'means':means,'stds':stds},indent=2))
if __name__=='__main__': main()
