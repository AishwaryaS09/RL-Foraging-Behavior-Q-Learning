# Reinforcement Learning-Based Animal Foraging Behavior Using Q-Learning

## Reinforcement Learning Assignment
**Project 23 :** Foraging behavior model -	A simulated animal choosing between food patches, with the learned policy compared against optimal foraging theory.

### Group Members
1. S.Aishwarya – 100523733051
2. D.Charan Teja – 100523733016


## Project Objective
Simulate an animal choosing between food patches and compare its learned foraging policy against optimal-foraging principles using Reinforcement Learning.

## Methodology
- Algorithm: Q-Learning
- Environment: Simulated food patches
- States: Current location and food availability
- Actions: Stay or switch food patches
- Rewards: Food collected minus movement costs
- Baselines: Random selection and heuristic policy

## Experimental Results

| Method | Average Reward |
|---|---:|
| Q-Learning | 52.81 |
| Random Policy | 44.82 |
| Heuristic Policy | 58.78 |

## Project Files
- `foraging_qlearning.py` – Python implementation
- `RL Assignment Report.pdf` – Project report
- `RL Assignment PPT.pptx` - PowerPoint presentation of Project methodology and findings
- `learning_curve.png` – Q-learning training graph
- `baseline_comparison.png` – Performance comparison

## Conclusion
The Q-learning agent achieved better performance than random selection, demonstrating the ability of reinforcement learning to learn useful foraging strategies. The heuristic baseline performed better in this simplified environment.

## How to Run
Install Python with NumPy and Matplotlib, then run:

`python foraging_qlearning.py`

## Academic Submission
Reinforcement Learning Assignment – Project 23 : Foraging behavior model.
