import random
import networkx as nx
import torch as T
from loguru import logger
import itertools
from collections import defaultdict

random.seed(42)


class Environment(object):
    def __init__(
        self,
        G: nx.Graph,
        nodes_exit: list = [],
        nodes_fire: list = [],
        device: T.device = T.device("cpu"),
        beta: float = 1.0,
        num_virtual_agents: int = 100,
        congestion_threshold: int = 5,
        reward_exit: float = 10000,
        reward_fire: float = 10000,
        reward_invalid: float = 5000,
        reward_valid: float = 1,
        reward_congestion: float = 300,
        agents_positions: list = [],
        test: bool = False,
    ) -> None:
        """
        Initializes the environment for a reinforcement learning agent in a graph-based setting.

        Args:
            G (nx.Graph): The graph representing the environment.
            nodes_exit (list, optional): List of nodes representing exit points. Defaults to [].
            nodes_fire (list, optional): List of nodes on fire, which should be avoided. Defaults to [].
            device (T.device, optional): Device to be used (CPU/GPU). Defaults to CPU.
            beta (float, optional): Scaling factor for certain calculations (e.g., rewards). Defaults to 1.0.
            num_virtual_agents (int, optional): Number of agents in the environment. Defaults to 10.
            congestion_threshold (int, optional): Maximum number of agents allowed in a node before congestion occurs. Defaults to 3.
        """
        self.device = device
        self.graph = G
        self.nodes_exit = nodes_exit
        self.nodes_fire = nodes_fire
        self.agent_positions = agents_positions
        self.beta = beta
        self.num_virtual_agents = num_virtual_agents
        self.congestion_threshold = congestion_threshold
        self.reward_exit = reward_exit
        self.reward_fire = reward_fire
        self.reward_invalid = reward_invalid
        self.reward_valid = reward_valid
        self.reward_congestion = reward_congestion

        self.state_visit_count = defaultdict(int)

        logger.info(f"Num agents: {self.num_virtual_agents}")

        node_list = list(self.graph.nodes)
        self.adj_mat = nx.adjacency_matrix(
            self.graph, nodelist=node_list
        ).todense()  # Generate adjacency matrix (dense format) based on the node list
        self.num_nodes = len(self.graph.nodes)

        self.n_actions = (
            self.num_nodes
        )  # Number of possible actions (equal to the number of nodes)
        self.obs_size = (
            self.num_nodes * 4
        )  # Observation size (based on node count, can be adjusted for additional state features)

        self.node_vector_end = [0.0] * self.num_nodes
        for end_node in self.nodes_exit:
            self.node_vector_end[end_node] = 1.0

        self.dict_node = self.create_dict()
        if not test:
            self.all_paths, self.critical_nodes = self.get_all_paths_to_exits()

        self.fire_node_counts = defaultdict(lambda: defaultdict(int))
        self.congestion_node_counts = defaultdict(lambda: defaultdict(int))
        self.state_visit_count = defaultdict(int)

    def get_all_paths_to_exits(self) -> tuple[dict, dict]:
        """
        Finds all simple paths from each node in the graph to each exit node,
        and identifies critical nodes that appear in all paths from each node to the exits.

        Returns:
            tuple:
                - all_paths: dict with source nodes as keys and list of paths to exits as values.
                - critical_nodes_by_source: dict with source nodes as keys and set of nodes that are critical for escape.
        """
        all_paths = {}
        critical_nodes_by_source = {}

        for node in self.graph.nodes:
            paths_from_node = []
            for exit_node in self.nodes_exit:
                if nx.has_path(self.graph, node, exit_node):
                    paths_generator = nx.shortest_simple_paths(
                        self.graph, source=node, target=exit_node
                    )
                    paths = list(itertools.islice(paths_generator, 3))
                    paths_from_node.extend(paths)
            all_paths[node] = paths_from_node

            # Cálculo dos nós críticos: interseção entre todos os caminhos
            if paths_from_node:
                path_sets = [set(p) for p in paths_from_node]
                critical_nodes = set.intersection(*path_sets)
                critical_nodes.discard(node)  # Remove o próprio nó de origem
                critical_nodes_by_source[node] = critical_nodes
            else:
                critical_nodes_by_source[node] = set()

            print(
                f"Nó {node}: {len(paths_from_node)} caminhos. Críticos: {critical_nodes_by_source[node]}"
            )

        return all_paths, critical_nodes_by_source

    def create_dict(self) -> dict:
        """
        Creates a mapping between node indices and their respective positions or attributes.

        Returns:
            dict: A dictionary mapping node indices to relevant information.
        """
        return {node: idx for idx, node in enumerate(self.graph.nodes)}

    def initialize_agents(self, agents_positions: list = []) -> list:
        """
        Initializes the positions of agents in the environment.

        Args:
            agents_positions (list, optional): Predefined agent positions. Defaults to an empty list.

        Returns:
            list: A list of agent starting positions.
        """
        # Initialize agent history to store movement over time
        self.agent_history = [[] for _ in range(self.num_virtual_agents)]

        # If predefined positions are provided, use them
        if agents_positions:
            return agents_positions

        # Otherwise, assign random or predefined starting positions
        return [self.define_start() for _ in range(self.num_virtual_agents)]

    def reward(self, current_node: int, new_node: int) -> tuple[float, bool, bool]:
        """
        Calculates the reward based on the transition from current_node to new_node.

        Args:
            current_node (int): The current node.
            new_node (int): The next node (where the agent moves).

        Returns:
            tuple[float, bool, bool]: The adjusted reward, a boolean indicating if the nodes are connected,
            and a boolean indicating if the agent has reached an exit node.
        """
        connected = False
        done = False

        # If the new node is on fire, the agent receives a severe penalty
        if new_node in self.nodes_fire:
            reward = -self.reward_fire
            connected = (
                new_node in self.graph[current_node]
            )  # Check if there is a connection with the current node

        # If the new node is connected to the current node
        elif new_node in self.graph[current_node]:
            if new_node in self.nodes_exit:  # If the agent reached an exit node
                reward = self.reward_exit
                done = True  # End of episode
            else:
                reward = -self.reward_valid * self.adj_mat.item(
                    (current_node, new_node)
                )  # Penalty based on graph distance

                # Additional penalty if the node has congestion
                num_virtual_agents_new_node = self.agent_positions.count(new_node)

                total_agents = len(self.agent_positions) if self.agent_positions else 1

                if num_virtual_agents_new_node >= self.congestion_threshold:
                    congestion_ratio = (
                        num_virtual_agents_new_node - self.congestion_threshold
                    ) / total_agents
                    reward -= self.reward_congestion * congestion_ratio
                    logger.info(
                        f"Reward: {3000 * congestion_ratio}, Congested node: {new_node}, Agents: {num_virtual_agents_new_node}, Ratio: {congestion_ratio} - {current_node} - {new_node}"
                    )

            connected = True

        else:  # If there is no connection between the nodes
            reward = -self.reward_invalid  # Penalty for invalid move
            connected = False

        return reward, connected, done

    def call_reward(self, current_node: int, action: int) -> tuple[float, bool, bool]:
        """
        Computes the adjusted reward using a discount factor based on the action taken.

        Args:
            current_node (int): The current node.
            action (int): The action taken, representing the next node.

        Returns:
            tuple[float, bool, bool]: The adjusted reward, a boolean indicating whether the nodes are connected,
            and a boolean indicating whether the episode has ended.
        """
        # Map the indices to the actual nodes in the graph
        mapped_current_node = self.dict_node[current_node]
        mapped_new_node = self.dict_node[action]

        # Calculate the reward, connection status, and episode completion
        reward, connected, done = self.reward(mapped_current_node, mapped_new_node)

        # Apply the discount factor to the reward
        discounted_reward = reward * self.beta

        return discounted_reward, connected, done

    def step(self, state: int, new_state: int) -> tuple[int, int, bool]:
        """
        Executes an action from a state and returns the next state,
        the reward, and a termination indicator.

        Args:
            state (int): The current state.
            action (int): The action taken.

        Returns:
            tuple[int, int, bool]: The next state, the associated reward, and
                                    a boolean indicating if the episode has ended.
        """
        # Get the reward, whether the nodes are connected, and if the episode is done
        reward, connected, done = self.call_reward(state, new_state)

        # If the nodes are not connected, stay in the current state
        if not connected:
            new_state = state

        return new_state, reward, done

    def define_start(self) -> int:
        """
        Defines a random starting node that is not an exit node,
        prioritizing the least accessed nodes.

        Returns:
            int: The chosen starting node.
        """
        if not hasattr(self, "node_access_count"):
            self.node_access_count = {node: 0 for node in range(self.num_nodes)}

        candidate_nodes = [
            node for node in range(self.num_nodes) if node not in self.nodes_exit
        ]

        min_access = min(self.node_access_count[node] for node in candidate_nodes)

        least_accessed_nodes = [
            node
            for node in candidate_nodes
            if self.node_access_count[node] == min_access
        ]

        chosen_node = random.choice(least_accessed_nodes)

        self.node_access_count[chosen_node] += 1

        return chosen_node

    def reset(
        self,
        previous_state: int = None,
        use_virtual_agents: bool = False,
        agents_virtuais_positions: list = [],
    ) -> int:
        """
        Resets the environment to its initial state, either based on a previous state
        or by selecting a new random state.

        Args:
            previous_state (int, optional): The previous state, if provided.
                                            If None, a new state is randomly chosen.
            use_virtual_agents (bool, optional): Whether to initialize virtual agents. Defaults to False.
            agents_virtuais_positions (list, optional): List of predefined positions for virtual agents. Defaults to an empty list.

        Returns:
            int: The initial state after the reset.
        """
        # Define the starting node: use previous_state if provided; otherwise, select a new one randomly.
        start = previous_state if previous_state is not None else self.define_start()

        # If virtual agents are not enabled, generate a scenario from the starting node.
        # Otherwise, use the provided positions for the virtual agents.
        if not use_virtual_agents:
            self.generate_scenario_from_node(start)

        else:
            self.agent_positions = agents_virtuais_positions

        self.bottleneck_vector = [
            self.agent_positions.count(node) / len(self.agent_positions)
            if (
                len(self.agent_positions)
                and self.agent_positions.count(node) >= self.congestion_threshold
            )
            else 0.0
            for node in range(self.num_nodes)
        ]

        return start

    def state_to_vector(self, current_node: int) -> list:
        """
        Converts the environment state (current and exit nodes) into a feature vector.

        Args:
            current_node (int): The current node in the graph.

        Returns:
            list: A feature vector representing the state.
        """
        current_node = int(current_node)  # Ensure the current node is an integer

        # One-hot encoding for the current node
        node_vector = [0.0] * self.num_nodes
        node_vector[current_node] = 1.0

        fire_vector = [0.0] * self.num_nodes
        for fire_node in self.nodes_fire:
            fire_vector[fire_node] = 1.0

        return node_vector + self.node_vector_end + fire_vector + self.bottleneck_vector

    # DEPRECATED: This function has been discontinued and is no longer used in the current workflow.
    # It was responsible for simulating the occurrence of fire on a node located along the path
    # between the current node and one of the exit nodes.
    def simulate_fire(self, state):
        """
        Simula a ocorrência de fogo em um nó localizado no caminho entre o nó x e os nós de saída.

        Parâmetros:
            state (int/str): O nó atual (estado) do agente.
        Retorna:
            O nó afetado por fogo (selecionado aleatoriamente dentre os candidatos)
            ou None, caso não ocorra a simulação de fogo.
        """
        candidate_nodes = set()

        for exit_node in self.nodes_exit:
            if nx.has_path(self.graph, state, exit_node):
                path = nx.shortest_path(self.graph, source=state, target=exit_node)
                if len(path) > 2:
                    candidate_nodes.update(path[1:-1])

        if not candidate_nodes:
            self.nodes_fire = []

        candidate_nodes = [
            node for node in candidate_nodes if node not in self.nodes_exit
        ]
        if random.random() < self.p_fire_nodes:
            self.nodes_fire = [random.choice(list(candidate_nodes))]
        else:
            self.nodes_fire = []

    # DEPRECATED: This function has been discontinued and is no longer used in the current workflow.
    # It was responsible for simulating the movement of virtual agents for exploration and congestion modeling purposes.
    def simulate_agents(self, state: int) -> list:
        """
        Simula o movimento de agentes virtuais, incentivando a exploração e introduzindo congestionamentos.

        Parâmetros:
            state (int): O estado atual do agente.

        Retorna:
            list: Posições atualizadas dos agentes.
        """
        if self.num_virtual_agents == 0 or state in self.nodes_exit:
            return []

        candidate_nodes = set()
        for exit_node in self.nodes_exit:
            if nx.has_path(self.graph, state, exit_node):
                path = nx.shortest_path(self.graph, source=state, target=exit_node)
                if len(path) > 2:
                    candidate_nodes.update(path[1:-1])  

        candidate_nodes = [
            node for node in candidate_nodes if node not in self.nodes_exit
        ]

        if not candidate_nodes:
            return []

        if not hasattr(self, "visit_history"):
            self.visit_history = {}

        if state not in self.visit_history:
            self.visit_history[state] = {node: 0 for node in candidate_nodes}
        else:
            for node in candidate_nodes:
                if node not in self.visit_history[state]:
                    self.visit_history[state][node] = 0

        less_visited = min(
            candidate_nodes,
            key=lambda n: (self.visit_history[state].get(n, 0), random.random()),
        )

        self.agent_positions = []
        if random.random() < self.p_congestion:
            for _ in range(self.num_virtual_agents):
                if (
                    len(candidate_nodes) > 1
                    and random.random() < self.p_random_neighbor
                ):
                    chosen_node = random.choice(candidate_nodes)
                else:
                    chosen_node = less_visited
                self.agent_positions.append(chosen_node)

            for chosen_node in self.agent_positions:
                if chosen_node in self.visit_history[state]:
                    self.visit_history[state][chosen_node] += 1

        return self.agent_positions

    def generate_scenario_from_node(self, state: int) -> dict:
        """
        Dynamically generates a simulated scenario based on the current state, which may include fire and/or
        congestion along parts of the path, depending on the visitation frequency of the state.

        Parameters:
            state (int): The current state of the agent.

        Returns:
            dict: A dictionary containing the selected path, fire nodes, and congestion nodes.
                Example:
                {
                    "path": [1, 2, 3],
                    "fire_nodes": [2],
                    "congestion_nodes": [3, 3, 3]
                }
        """

        if state not in self.all_paths or not self.all_paths[state]:
            return {"path": [], "fire_nodes": [], "congestion_nodes": []}

        path = random.choice(self.all_paths[state])

        fire_nodes = []
        congestion_nodes = []

        self.state_visit_count[state] += 1

        if self.state_visit_count[state] <= 2:
            scenario_type = 0
        else:
            scenario_type = random.choice(
                [0, 1, 2, 3]
            )  # 0: wo obstacles, 1: fire, 2: congestion, 3: both

        self.num_virtual_agents = random.randint(0, 20)

        other_paths = [p for p in self.all_paths[state] if p != path]
        has_other_paths = len(other_paths) > 0

        fire_path = path
        congestion_path = random.choice(other_paths) if has_other_paths else path

        critical_nodes = self.critical_nodes.get(state, set())

        if scenario_type in [1, 3]:  # Fogo
            fire_intermediates = [
                n
                for n in fire_path[1:-1]
                if n not in self.nodes_exit and n not in critical_nodes
            ]
            if fire_intermediates:
                fire_node = self.weighted_choice_least_used(
                    fire_intermediates, self.fire_node_counts, state
                )
                fire_nodes = [fire_node]
                self.fire_node_counts[state][fire_node] += 1

        if scenario_type in [2, 3]:
            congestion_intermediates = [
                n
                for n in congestion_path[1:-1]
                if n not in self.nodes_exit and n not in fire_nodes
            ]
            if congestion_intermediates:
                congested_node = self.weighted_choice_least_used(
                    congestion_intermediates, self.congestion_node_counts, state
                )
                self.congestion_node_counts[state][congested_node] += 1

                min_agents = max(self.congestion_threshold, 1)
                self.num_virtual_agents = random.randint(min_agents, 20)
                congestion_nodes = [congested_node] * self.num_virtual_agents

        self.agent_positions = congestion_nodes
        self.nodes_fire = fire_nodes

        return {
            "path": path,
            "fire_nodes": fire_nodes,
            "congestion_nodes": congestion_nodes,
        }

    @staticmethod
    def weighted_choice_least_used(nodes, counter_by_state, state):
        """
        Performs a weighted random choice among the available nodes, favoring those
        that have been used less frequently based on a usage counter per state.

        Parameters:
            nodes (list): List of candidate nodes.
            counter_by_state (dict): Dictionary with usage counters of nodes, indexed by state.
            state (int): Current state being considered.

        Returns:
            int or None: A selected node from the list with the lowest usage frequency, or None if the list is empty.
        """
        if not nodes:
            return None
        weights = [1 / (1 + counter_by_state[state][n]) for n in nodes]
        return random.choices(nodes, weights=weights, k=1)[0]
