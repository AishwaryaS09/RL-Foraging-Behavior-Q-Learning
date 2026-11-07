# Reinforcement Learning-Based Animal Foraging Behavior Using Q-Learning

## Reinforcement Learning – Homework 2
**Project Number:** 23

### Group Members
1. Full Name – Roll Number
2. Full Name – Roll Number
3. Full Name – Roll Number

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
- `Project23_Report.pdf` – Project report
- PowerPoint presentation – Project methodology and findings
- `learning_curve.png` – Q-learning training graph
- `baseline_comparison.png` – Performance comparison

## Conclusion
The Q-learning agent achieved better performance than random selection, demonstrating the ability of reinforcement learning to learn useful foraging strategies. The heuristic baseline performed better in this simplified environment.

## How to Run
Install Python with NumPy and Matplotlib, then run:

`python foraging_qlearning.py`

## Academic Submission
Reinforcement Learning Homework 2 – Project 23.
