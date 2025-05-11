# Influence Maximization with Haters

This project simulates influence spread in a social network graph under budget constraints, using a variant of the Independent Cascade model that accounts for "haters"—nodes that suppress influence spread to their neighbors.

## Overview

Given a social network represented as a graph, the goal is to select a set of influencers within a fixed budget such that their total influence spread over several rounds (default is 6) is maximized.

The algorithm progressively selects influencer nodes based on their potential to affect their surroundings, while accounting for both their cost and network context.

## Infection Model

We use a modified Independent Cascade (IC) model:

- Each node can influence its neighbors **once** per round.
- The base influence probability is defined as `P_BASE = 0.2`.
- A node is influenced if at least one of its neighbors influences it.
- The influence probability from a neighbor `u` to a candidate node `v` is reduced based on `v`'s neighboring haters.

## Features
- Hater-aware IC model
- Greedy influencer selection
- Simulation-based evaluation
- Budget-aware influencer selection

## Files
- `main.py` — Main simulation script
- The following files are auto-generated when running the project:
  - `NoseBook_friendships.csv` — Social graph edges
  - `haters.csv` — Hater nodes and weights
  - `costs.csv` — Cost of selecting each node as an influencer
  - `submission.csv` — Final selected influencer list


## Requirements
This project requires Python 3.7+ and the following packages:

- `networkx`
- `numpy`
- `pandas`
