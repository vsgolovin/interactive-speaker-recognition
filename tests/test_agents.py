"""
Simple tests for simple agents:
1. correct shape
2. valid values
3. no repetitions
"""

import pytest
import numpy as np
import torch
from isr.simple_agents import HeuristicAgent, RandomAgent


@pytest.mark.parametrize("bs,nw", [(1, 1), (8, 2), (16, 3)])
def test_heuristic_agent(bs: int, nw: int):
    word_scores = np.array([0.0, 1.0, 5.0, 2.0, 0.5])
    agent = HeuristicAgent(word_scores, k=3)
    samples = agent.sample(num_envs=bs, num_words=nw)
    unique_words = set(torch.unique(samples).numpy())
    assert samples.size() == torch.Size((bs, nw)) \
        and unique_words.issubset({1, 2, 3}) \
        and all(len(torch.unique(row)) == nw for row in samples)


@pytest.mark.parametrize("bs,nw,v", [(1, 1, 1), (8, 3, 10), (4, 5, 6)])
def test_random_agent(bs: int, nw: int, v: int):
    agent = RandomAgent(total_words=v)
    samples = agent.sample(bs, nw)
    unique_words = set(torch.unique(samples).numpy())
    assert samples.size() == torch.Size((bs, nw)) \
        and unique_words.issubset(range(v)) \
        and all(len(torch.unique(row)) == nw for row in samples)
