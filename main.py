"""
Influence Maximization Simulator
--------------------------------
This script simulates influence propagation in a social network using
a variant of the Independent Cascade (IC) model with the presence of "haters"
who reduce the influence probability of their neighbors.

Developed for academic purposes.
"""

import csv
import random
import networkx as nx
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

BUDGET = 1500
ROUNDS = 6
P_BASE = 0.2

# Default file paths
FRIENDSHIPS_FILENAME = 'NoseBook_friendships.csv'
HATERS_FILENAME = 'haters.csv'
COSTS_FILENAME = 'costs.csv'

def load_graph(path):
	"""
    Load an undirected social graph from a CSV file.

    Each row in the CSV should have two columns: 'user' and 'friend',
    representing a bidirectional friendship between two nodes.

    Returns:
        A NetworkX Graph object.
    """
	try:
		df = pd.read_csv(path)
		return nx.from_pandas_edgelist(df, source='user', target='friend')
	except FileNotFoundError:
		print(f"Graph file not found: {path}")
	except Exception as e:
		print(f"Failed to load graph from {path}: {e}")
	return None

def load_haters(path):
	"""
    Load a dictionary of hater nodes and their influence-reduction weights.

    Each row in the CSV should contain:
        user_id, weight

    Returns:
        A dictionary {user_id: weight}
    """
	try:
		df = pd.read_csv(path)
		return dict(zip(df['user_id'].astype(int), df['weight'].astype(float)))
	except FileNotFoundError:
		print(f"Hater file not found: {path}")
	except Exception as e:
		print(f"Failed to load haters from {path}: {e}")
	return None

def load_costs(path):
	"""
    Load the cost associated with selecting each node as an influencer.

    Each row in the CSV should contain:
        user_id, cost

    Returns:
        A dictionary {user_id: cost}
    """
	try:
		df = pd.read_csv(path)
		return dict(zip(df['user_id'].astype(int), df['cost'].astype(float)))
	except FileNotFoundError:
		print(f"Cost file not found: {path}")
	except Exception as e:
		print(f"Failed to load costs from {path}: {e}")
	return None

def simulate_influence_spread(graph, seed_nodes, haters, p_base=0.2, rounds=6):
	"""
    Simulate influence spread over the network using a modified Independent Cascade model.

    Nodes may become influenced by their neighbors based on a base probability,
    which is reduced according to the presence of hater neighbors.

    Args:
        graph (nx.Graph): The social network.
        seed_nodes (list): Initial influencer node IDs.
        haters (dict): Map of hater node IDs to their influence-suppression weights.
        p_base (float): Base probability of influence.
        rounds (int): Number of simulation steps.

    Returns:
        Total number of influenced nodes at the end of the simulation.
    """
	if graph is None or seed_nodes is None or haters is None:
		return 0

	haters_set = set(haters)
	active = set(seed_nodes) - haters_set
	all_nodes = list(graph.nodes())

	for _ in range(rounds):
		new_activations = set()

		for node in all_nodes:
			if node in active or node in haters_set:
				continue

			neighbors = set(graph.neighbors(node))
			influencers = [n for n in neighbors if n in active and n not in haters_set]

			if not influencers:
				continue

			# Compute hater suppression
			hater_neighbors = [n for n in neighbors if n in haters_set]
			suppression = np.prod([1 - haters[h] for h in hater_neighbors]) if hater_neighbors else 1.0
			p_effective = p_base * suppression

			prob_not_activated = np.prod([1 - p_effective for _ in influencers])
			if random.random() < (1 - prob_not_activated):
				new_activations.add(node)

		active.update(new_activations)

	return len(active)


def export_influencer_selection(nodes, costs, haters, filename="submission.csv", budget=1500):
	"""
    Export a list of influencer node IDs to a CSV file after validation.

    Checks:
    - No duplicate nodes
    - All nodes exist in cost data
    - None are haters
    - Total cost is within budget

    Returns:
        True if file was written successfully, False otherwise.
    """
	if not nodes or not isinstance(nodes, list):
		print("Invalid influencer list.")
		return False

	seen = set()
	total_cost = 0.0
	for node in nodes:
		if node in seen:
			print(f"Duplicate node detected: {node}")
			return False
		if node in haters:
			print(f"Cannot include hater node: {node}")
			return False
		if node not in costs:
			print(f"No cost data for node: {node}")
			return False

		seen.add(node)
		total_cost += costs[node]

	if total_cost > budget:
		print(f"Total cost {total_cost:.2f} exceeds budget {budget}.")
		return False

	try:
		with open(filename, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['user_id'])
			for node in sorted(nodes):
				writer.writerow([node])
		print(f"Influencer list exported to {filename} (Total cost: {total_cost:.2f})")
		return True
	except Exception as e:
		print(f"Failed to write to {filename}: {e}")
		return False


def compute_node_easiness(graph, haters):
	"""
    Computes easiness score for each node:
    how easy it is to infect that node itself.
    """
	easiness = {}
	not_haters = set(graph.nodes()) - set(haters.keys())

	for node in not_haters:
		neighbors = set(graph.neighbors(node))

		healthy_neighbors = [n for n in neighbors if n not in haters]
		hater_neighbors = [n for n in neighbors if n in haters]

		num_healthy = len(healthy_neighbors)

		# Compute product over hater neighbors
		product = 1.0
		for hater in hater_neighbors:
			product *= (1 - haters[hater])

		base_value = 0.2 * product

		# Infection probability:
		score = 1 - (1 - base_value) ** num_healthy

		easiness[node] = score

	return easiness


def compute_propagation_potential(graph, easiness, haters, infected):
	"""
    Computes the sum of easiness of neighbors for each node.
    Exclude already infected nodes and haters.
    """
	propagation = {}
	not_haters = set(graph.nodes()) - set(haters.keys())

	for node in not_haters:
		if node in infected:
			continue  # Already infected

		neighbors = set(graph.neighbors(node))

		score = 0
		for neighbor in neighbors:
			if neighbor not in haters and neighbor not in infected:
				score += easiness.get(neighbor, 0)

		propagation[node] = score

	return propagation


def simulate_current_infection(graph, selected, haters, rounds=ROUNDS, p_base=P_BASE):
	"""
    Simulates infection process starting from selected nodes,
    considering haters, for a few rounds.
    """
	haters_set = set(haters.keys())
	influenced_nodes = set(selected) - haters_set
	all_nodes = list(graph.nodes())

	for _ in range(1, rounds + 1):
		newly_influenced = set()
		for candidate in all_nodes:
			if candidate in influenced_nodes or candidate in haters_set:
				continue

			try:
				candidate_neighbors = set(graph.neighbors(candidate))
			except nx.NetworkXError:
				continue

			influencing_neighbors = {
				u for u in candidate_neighbors
				if u in influenced_nodes and u not in haters_set
			}

			if not influencing_neighbors:
				continue

			hater_neighbors = {h for h in candidate_neighbors if h in haters_set}
			anti_influence = 1.0
			for hater_node in hater_neighbors:
				anti_influence *= (1.0 - haters[hater_node])

			prob_not_infected_by_any = 1.0
			for u in influencing_neighbors:
				p_effective = p_base * anti_influence
				prob_not_infected_by_any *= (1.0 - p_effective)

			prob_infected = 1.0 - prob_not_infected_by_any

			if random.random() < prob_infected:
				newly_influenced.add(candidate)

		influenced_nodes.update(newly_influenced)
	return influenced_nodes


def select_influencers_progressively(graph, costs, haters, budget=1500):
	"""
    Select influencers one-by-one based on updated propagation potentials,
    """
	easiness = compute_node_easiness(graph, haters)
	infected = set(haters.keys())
	selected = []
	remaining_budget = budget

	print("\nSelecting influencers...")

	while True:
		propagation = compute_propagation_potential(graph, easiness, haters, infected)

		if not propagation:
			break

		# Filter only nodes that are affordable
		affordable_nodes = {node: score for node, score in propagation.items()
							if costs.get(node, float('inf')) <= remaining_budget}

		if not affordable_nodes:
			break  # No affordable nodes left

		# Here we will also consider the cost of each node:
		best_node = max(affordable_nodes, key=lambda x: affordable_nodes[x] / (costs.get(x, 1) / 200))
		cost = costs.get(best_node, float('inf'))

		if cost > remaining_budget:
			del propagation[best_node]
			if all(costs.get(n, float('inf')) > remaining_budget for n in propagation):
				break
			continue

		selected.append(best_node)
		remaining_budget -= cost

		# Update infection status
		infected = simulate_current_infection(graph, selected, haters)

		print(f"Selected {best_node}, Remaining Budget: {remaining_budget:.2f}")

		if remaining_budget <= 0:
			break

	print("\nFinished selecting influencers.\n")

	return selected

def generate_fake_data():
	# Generate a large synthetic graph (e.g., 3000 nodes)
	num_nodes = 3000
	edges_per_node = 5

	G = nx.generators.random_graphs.barabasi_albert_graph(num_nodes, edges_per_node)

	# Save friendships
	edges = nx.to_pandas_edgelist(G)
	edges.columns = ['user', 'friend']
	edges.to_csv("NoseBook_friendships.csv", index=False)

	# Assign random hater weights to ~5% of nodes
	all_nodes = list(G.nodes())
	haters = np.random.choice(all_nodes, size=int(0.05 * num_nodes), replace=False)
	weights = np.random.uniform(0.2, 0.9, size=len(haters))
	pd.DataFrame({'user_id': haters, 'weight': weights}).to_csv("haters.csv", index=False)

	# Assign random costs between 50–200 to all nodes
	costs = pd.DataFrame({
		'user_id': all_nodes,
		'cost': np.random.randint(50, 200, size=len(all_nodes))
	})
	costs.to_csv("costs.csv", index=False)

	print("Synthetic data generated: NoseBook_friendships.csv, haters.csv, costs.csv")

if __name__ == '__main__':
	print('--- Influence Maximization Simulation ---')
	generate_fake_data()
	graph = load_graph(FRIENDSHIPS_FILENAME)
	haters = load_haters(HATERS_FILENAME)
	costs = load_costs(COSTS_FILENAME)

	selections = []
	scores = []
	for i in range(20):
		influencers = select_influencers_progressively(graph, costs, haters)
		selections.append(influencers)
		sims = [simulate_influence_spread(graph, influencers, haters) for _ in range(100)]
		avg_score = np.mean(sims)
		scores.append(avg_score)
		print(f"Run #{i+1}: Avg Influence = {avg_score:.2f}, Cost = {sum(costs[n] for n in influencers):.2f}")

	best_idx = int(np.argmax(scores))
	best_influencers = selections[best_idx]
	best_score = scores[best_idx]
	export_influencer_selection(best_influencers, costs, haters)
	print(f"\nBest Result: Run #{best_idx+1}, Avg Influence = {best_score:.2f}")
