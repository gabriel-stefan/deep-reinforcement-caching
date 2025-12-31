#!/usr/bin/env python3
import sys
import os
import time
import argparse
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents import get_policy


def evaluate_policy(env, policy, name, steps, verbose=True):
    if verbose:
        print(f"evaluating {name}...")
    
    obs, _ = env.reset()
    policy.reset()
    start = time.time()
    
    for step in range(steps):
        action = policy.select_multinode_action(obs, env)
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    
    elapsed = time.time() - start
    metrics = env.get_metrics()
    metrics['elapsed_time'] = elapsed
    metrics['steps'] = step + 1
    
    if verbose:
        print(f"  {metrics['hit_rate']:.2%} hit rate, {elapsed:.1f}s")
    
    return metrics


def print_results(results):
    print(f"\n{'policy':<12} {'hit rate':>10} {'byte hit':>10} {'time':>8}")
    print("-" * 42)
    
    for name, m in results:
        print(f"{name:<12} {m['hit_rate']:>9.2%} {m['byte_hit_rate']:>9.2%} {m['elapsed_time']:>7.1f}s")
    
    best = max(results, key=lambda x: x[1]['hit_rate'])
    print(f"\nbest: {best[0]} ({best[1]['hit_rate']:.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--cache-mb', type=float, default=1.0)
    parser.add_argument('--regional-mb', type=float, default=None)
    parser.add_argument('--candidates', type=int, default=20)
    parser.add_argument('--policies', nargs='+', 
                        default=['lru', 'lfu', 'fifo', 'size', 'gdsf', 'hyper', 'random'],
                        choices=['random', 'lru', 'lfu', 'fifo', 'size', 'gdsf', 'hyper'])
    parser.add_argument('--data', type=str, default='data/processed/consistent_trace.csv')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    
    # regional defaults to 2x edge
    regional_mb = args.regional_mb if args.regional_mb else args.cache_mb * 2
    
    print(f"cache: edge={args.cache_mb}mb, regional={regional_mb}mb")
    print(f"steps: {args.steps}, policies: {', '.join(args.policies)}")
    
    loader = DataLoader(args.data, split_ratio=0.7, mode=args.mode)
    results = []
    
    for policy_name in args.policies:
        loader.reset()
        topology = create_simple_hierarchy(
            edge_capacity_mb=args.cache_mb,
            regional_capacity_mb=regional_mb
        )
        env = CacheEnv(loader, topology=topology, k_candidates=args.candidates)
        policy = get_policy(policy_name, n_candidates=args.candidates)
        metrics = evaluate_policy(env, policy, policy_name, args.steps, not args.quiet)
        results.append((policy_name, metrics))
    
    print_results(results)


if __name__ == '__main__':
    main()
