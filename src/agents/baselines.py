import numpy as np
from abc import ABC
from typing import Dict
from src.environments.core.cache_storage import CacheItem


class BaselinePolicy(ABC):
    def __init__(self, n_candidates: int):
        self.n_candidates = n_candidates
        self.name = "base"
    
    def reset(self):
        pass
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        raise NotImplementedError
    
    def _get_skip_action(self, env) -> int:
        return env.num_nodes * env.k_candidates
    
    def _get_node_action_offset(self, env, node_idx: int) -> int:
        return node_idx * env.k_candidates
    
    def _find_item_in_candidates(self, env, node_id: str, target_url: str) -> int:
        candidates = env._candidates.get(node_id, [])
        for i, item in enumerate(candidates[:env.k_candidates]):
            if item.url == target_url:
                return i
        return -1
    
    def _direct_evict_and_cache(self, env, node_idx: int, node_id: str, 
                                 evict_url: str, new_url: str, size: int, 
                                 timestamp: float) -> int:
        node = env.topology.get_node(node_id)
        node.evict(evict_url)
        node.add(new_url, size, timestamp)
        return self._get_skip_action(env)


class RandomPolicy(BaselinePolicy):
    # evicts random item from all cache items
    
    def __init__(self, n_candidates: int, skip_probability: float = 0.1):
        super().__init__(n_candidates)
        self.name = "random"
        self.skip_prob = skip_probability
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        if env.request is None:
            return self._get_skip_action(env)
        
        url, size, timestamp = env.request[0], env.request[1], env.request[2]
        skip_action = self._get_skip_action(env)
        
        if np.random.random() < self.skip_prob:
            return skip_action
        
        # check for hit
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.cache.contains(url):
                return skip_action
        
        # check for free space
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.get_free_space() >= size:
                return self._get_node_action_offset(env, node_idx) + 0
        
        # pick random item from all items across all nodes
        all_items = []
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            for item in node.cache.get_all_items():
                all_items.append((node_idx, node_id, item))
        
        if not all_items:
            return skip_action
        
        # random eviction
        best_node_idx, best_node_id, best_item = all_items[np.random.randint(len(all_items))]
        
        candidate_idx = self._find_item_in_candidates(env, best_node_id, best_item.url)
        if candidate_idx >= 0:
            return self._get_node_action_offset(env, best_node_idx) + candidate_idx
        else:
            return self._direct_evict_and_cache(
                env, best_node_idx, best_node_id,
                best_item.url, url, size, timestamp
            )


class LRUPolicy(BaselinePolicy):
    # evicts least recently used item
    
    def __init__(self, n_candidates: int):
        super().__init__(n_candidates)
        self.name = "lru"
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        if env.request is None:
            return self._get_skip_action(env)
            
        url, size, timestamp = env.request[0], env.request[1], env.request[2]
        skip_action = self._get_skip_action(env)
        
        # check for hit
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.cache.contains(url):
                return skip_action
        
        # check for free space
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.get_free_space() >= size:
                return self._get_node_action_offset(env, node_idx) + 0
        
        # find lru item
        best_node_idx = None
        best_node_id = None
        best_item = None
        oldest_time = float('inf')
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            for item in node.cache.get_all_items():
                if item.last_access < oldest_time:
                    oldest_time = item.last_access
                    best_node_idx = node_idx
                    best_node_id = node_id
                    best_item = item
        
        if best_item is None:
            return skip_action
        
        candidate_idx = self._find_item_in_candidates(env, best_node_id, best_item.url)
        if candidate_idx >= 0:
            return self._get_node_action_offset(env, best_node_idx) + candidate_idx
        else:
            return self._direct_evict_and_cache(
                env, best_node_idx, best_node_id, 
                best_item.url, url, size, timestamp
            )


class LFUPolicy(BaselinePolicy):
    # evicts least frequently used item (lru tiebreaker)
    
    def __init__(self, n_candidates: int):
        super().__init__(n_candidates)
        self.name = "lfu"
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        if env.request is None:
            return self._get_skip_action(env)
            
        url, size, timestamp = env.request[0], env.request[1], env.request[2]
        skip_action = self._get_skip_action(env)
        
        # check for hit
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.cache.contains(url):
                return skip_action
        
        # check for free space
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.get_free_space() >= size:
                return self._get_node_action_offset(env, node_idx) + 0
        
        # find lfu item with lru tiebreaker
        best_node_idx = None
        best_node_id = None
        best_item = None
        min_freq = float('inf')
        oldest_time = float('inf')
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            for item in node.cache.get_all_items():
                if (item.frequency < min_freq or 
                    (item.frequency == min_freq and item.last_access < oldest_time)):
                    min_freq = item.frequency
                    oldest_time = item.last_access
                    best_node_idx = node_idx
                    best_node_id = node_id
                    best_item = item
        
        if best_item is None:
            return skip_action
        
        candidate_idx = self._find_item_in_candidates(env, best_node_id, best_item.url)
        if candidate_idx >= 0:
            return self._get_node_action_offset(env, best_node_idx) + candidate_idx
        else:
            return self._direct_evict_and_cache(
                env, best_node_idx, best_node_id,
                best_item.url, url, size, timestamp
            )


class FIFOPolicy(BaselinePolicy):
    # evicts oldest inserted item (first in first out)
    
    def __init__(self, n_candidates: int):
        super().__init__(n_candidates)
        self.name = "fifo"
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        if env.request is None:
            return self._get_skip_action(env)
            
        url, size, timestamp = env.request[0], env.request[1], env.request[2]
        skip_action = self._get_skip_action(env)
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.cache.contains(url):
                return skip_action
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.get_free_space() >= size:
                return self._get_node_action_offset(env, node_idx) + 0
        
        best_node_idx = None
        best_node_id = None
        best_item = None
        oldest_insert = float('inf')
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            for item in node.cache.get_all_items():
                if item.first_access < oldest_insert:
                    oldest_insert = item.first_access
                    best_node_idx = node_idx
                    best_node_id = node_id
                    best_item = item
        
        if best_item is None:
            return skip_action
        
        candidate_idx = self._find_item_in_candidates(env, best_node_id, best_item.url)
        if candidate_idx >= 0:
            return self._get_node_action_offset(env, best_node_idx) + candidate_idx
        else:
            return self._direct_evict_and_cache(
                env, best_node_idx, best_node_id,
                best_item.url, url, size, timestamp
            )


class LargestFirstPolicy(BaselinePolicy):
    # evicts largest item first (greedy-dual-size heuristic)
    
    def __init__(self, n_candidates: int):
        super().__init__(n_candidates)
        self.name = "size"
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        if env.request is None:
            return self._get_skip_action(env)
            
        url, size, timestamp = env.request[0], env.request[1], env.request[2]
        skip_action = self._get_skip_action(env)
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.cache.contains(url):
                return skip_action
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.get_free_space() >= size:
                return self._get_node_action_offset(env, node_idx) + 0
        
        best_node_idx = None
        best_node_id = None
        best_item = None
        max_size = 0
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            for item in node.cache.get_all_items():
                if item.size > max_size:
                    max_size = item.size
                    best_node_idx = node_idx
                    best_node_id = node_id
                    best_item = item
        
        if best_item is None:
            return skip_action
        
        candidate_idx = self._find_item_in_candidates(env, best_node_id, best_item.url)
        if candidate_idx >= 0:
            return self._get_node_action_offset(env, best_node_idx) + candidate_idx
        else:
            return self._direct_evict_and_cache(
                env, best_node_idx, best_node_id,
                best_item.url, url, size, timestamp
            )


class GDSFPolicy(BaselinePolicy):
    # greedy-dual-size-frequency (used by squid proxy)
    # evicts item with lowest priority = frequency / size
    # keeps small popular items, evicts large unpopular ones
    
    def __init__(self, n_candidates: int):
        super().__init__(n_candidates)
        self.name = "gdsf"
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        if env.request is None:
            return self._get_skip_action(env)
            
        url, size, timestamp = env.request[0], env.request[1], env.request[2]
        skip_action = self._get_skip_action(env)
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.cache.contains(url):
                return skip_action
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.get_free_space() >= size:
                return self._get_node_action_offset(env, node_idx) + 0
        
        # find item with lowest priority (frequency / size)
        best_node_idx = None
        best_node_id = None
        best_item = None
        min_priority = float('inf')
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            for item in node.cache.get_all_items():
                # priority = frequency / size (higher = more valuable)
                priority = item.frequency / max(item.size, 1)
                if priority < min_priority:
                    min_priority = priority
                    best_node_idx = node_idx
                    best_node_id = node_id
                    best_item = item
        
        if best_item is None:
            return skip_action
        
        candidate_idx = self._find_item_in_candidates(env, best_node_id, best_item.url)
        if candidate_idx >= 0:
            return self._get_node_action_offset(env, best_node_idx) + candidate_idx
        else:
            return self._direct_evict_and_cache(
                env, best_node_idx, best_node_id,
                best_item.url, url, size, timestamp
            )


class HyperbolicPolicy(BaselinePolicy):
    # hyperbolic caching (used by varnish cdn)
    # evicts item with lowest priority = frequency / age
    # balances recency with popularity
    
    def __init__(self, n_candidates: int):
        super().__init__(n_candidates)
        self.name = "hyper"
    
    def select_multinode_action(self, observation: np.ndarray, env) -> int:
        if env.request is None:
            return self._get_skip_action(env)
            
        url, size, timestamp = env.request[0], env.request[1], env.request[2]
        skip_action = self._get_skip_action(env)
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.cache.contains(url):
                return skip_action
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            if node.get_free_space() >= size:
                return self._get_node_action_offset(env, node_idx) + 0
        
        # find item with lowest priority (frequency / age)
        best_node_idx = None
        best_node_id = None
        best_item = None
        min_priority = float('inf')
        
        for node_idx, node_id in enumerate(env.nodes):
            node = env.topology.get_node(node_id)
            for item in node.cache.get_all_items():
                # age = time since first cached
                age = max(timestamp - item.first_access, 1)
                # priority = frequency / age (higher = more valuable)
                priority = item.frequency / age
                if priority < min_priority:
                    min_priority = priority
                    best_node_idx = node_idx
                    best_node_id = node_id
                    best_item = item
        
        if best_item is None:
            return skip_action
        
        candidate_idx = self._find_item_in_candidates(env, best_node_id, best_item.url)
        if candidate_idx >= 0:
            return self._get_node_action_offset(env, best_node_idx) + candidate_idx
        else:
            return self._direct_evict_and_cache(
                env, best_node_idx, best_node_id,
                best_item.url, url, size, timestamp
            )


# registry
POLICIES: Dict[str, type] = {
    'random': RandomPolicy,
    'lru': LRUPolicy,
    'lfu': LFUPolicy,
    'fifo': FIFOPolicy,
    'size': LargestFirstPolicy,
    'gdsf': GDSFPolicy,
    'hyper': HyperbolicPolicy,
}


def get_policy(name: str, n_candidates: int = 20) -> BaselinePolicy:
    name = name.lower()
    if name not in POLICIES:
        raise ValueError(f"unknown policy '{name}'. available: {list(POLICIES.keys())}")
    return POLICIES[name](n_candidates)
