# Caching with Deep Reinforcement Learning

## Project Overview

This project explores the application of Deep Reinforcement Learning to optimize content placement in a hierarchical Content Delivery Network. 

The project intentionally separates the caching problem into two components:

1. **Eviction**: Handled by heuristic LFU, as it is optimal for the given workload.
2. **Placement**: Learned with Deep Reinforcement Learning agents that decide where to store content (Edge, Regional, or Skip).


## Placement Baselines (1 Million Requests)

| Policy                 | Hit Rate   | Reward (Latency) |
| :--------------------- | :--------- | :--------------- |
| **SizeSplit (Median)** | **52.06%** | **4,472,646**    |
| PercentileSplit (P90)  | 46.62%     | 4,275,183        |
| EdgeFirst              | 43.10%     | 4,017,288        |
| Probabilistic (10%)    | 42.67%     | 3,721,054        |

## Environment

The CDN is modeled as a three-tier hierarchical network:

* **Edge (Tier 0)**: Low latency, high bandwidth, limited storage.
* **Regional (Tier 1)**: Intermediate latency and capacity.
* **Origin**: Unlimited capacity but highest latency and lowest bandwidth.

### Placement Action Space

The RL agent chooses one of three discrete actions for each request:

* **Cache at Edge**
* **Cache at Regional**
* **Skip** (serve from Origin without caching)

### Reward Function

Rewards are based on relative latency savings:

```
R = ((L_origin - L_actual) / L_origin) * 10
```

## Implemented Agents

* **DQN** (Stable Baselines3)
* **PPO** (Stable Baselines3)
* **Discrete SAC** (Custom implementation and adapted from CleanRL)

## Final Comparison

Best configuration from each category, sorted by Hit Rate:

| Rank | Method                         | Type      | Hit Rate |
| :--: | :----------------------------- | :-------- | :------- |
|   1  | SizeSplit (Median)             | Heuristic | 52.06%   |
|   2  | PPO (Reduced Clip)             | RL        | 51.97%   |
|   3  | Discrete SAC (CleanRL, stable) | RL        | 51.79%   |
|   4  | DQN (Best configuration)       | RL        | 51.70%   |
|   5  | Discrete SAC (Custom)          | RL        | 49.63%   |
|   6  | PercentileSplit (P90)          | Heuristic | 46.62%   |
|   7  | EdgeFirst                      | Heuristic | 43.10%   |
|   8  | Probabilistic (10%)            | Heuristic | 42.67%   |

## Data

Uses Wikipedia request traces from *Wikipedia Workload Analysis for Decentralized Hosting* by Guido Urdaneta, Guillaume Pierre, and Maarten van Steen.

## Conclusions

* On this environment, Reinforcement Learning is most effective when applied to placement, not eviction.
* PPO and Discrete SAC produced the most stable and balanced policies, while DQN showed to be unstable.

