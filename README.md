# Caching with Deep Reinforcement Learning

A simulation environment for evaluating cache replacement policies using real-world Wikipedia traces.


## Baseline Results

Wikipedia trace, 50k steps, 1MB edge + 2MB regional cache:

| Policy | Hit Rate | Byte Hit Rate | Time |
|--------|----------|---------------|------|
| *SIZE* | *66.55%* | 56.42% | 388.9s |
| Random | 64.44% | 56.57% | 314.4s |
| LFU | 47.93% | 45.96% | 312.3s |
| Hyperbolic | 47.27% | 45.15% | 318.1s |
| GDSF | 44.60% | 40.29% | 327.1s |
| FIFO | 41.61% | 36.38% | 314.0s |
| LRU | 41.54% | 39.08% | 317.3s |


## Implemented Policies

- *LRU* - Least Recently Used
- *LFU* - Least Frequently Used  
- *FIFO* - First In First Out
- *SIZE* - Evict largest item first
- *GDSF* - Greedy Dual Size Frequency
- *Hyperbolic* - Hyperbolic caching (frequency × recency / size)
- *Random* - Random eviction

## Data

Uses the Wikipedia traces from Wikipedia Workload Analysis for Decentralized Hosting by Guido Urdaneta, Guillaume Pierre, Maarten van Steen.