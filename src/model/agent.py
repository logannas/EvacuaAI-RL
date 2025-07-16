import time
import random
import networkx as nx
import numpy as np
import torch as T
import torch.nn as nn
import io
from tqdm import tqdm
from collections import deque
from loguru import logger
import matplotlib.pyplot as plt
from collections import Counter

from src.model.network import Network
from src.model.environment import Environment


class Agent:
    def __init__(
        self,
        graph: nx.Graph,
        nodes_fire: list = [],
        nodes_exit: list = [],
        hyperparameters: dict = {},
        model_pt: bytes | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Agent with the given parameters.

        Args:
            graph (nx.Graph): The environment graph.
            nodes_fire (list): List of nodes on fire.
            nodes_exit (list): List of exit nodes.
            transfer_learning (str | None): Path to pre-trained model weights.
            hyperparameters (dict): Dictionary of hyperparameters.
            save_model_path (str | None): Path to save the trained model.
            verbose (bool): If True, enables detailed logging.
            num_agents (int): Number of agents in the environment.
        """
        self.verbose = verbose
        logger.info(f"Hyperparameters: {hyperparameters}")

        # Set device for computations (GPU if available, otherwise CPU)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.hyperparameters = hyperparameters
        self.min_replay_size = int(self.hyperparameters.get("buffer_size") * 0.25)

        # Initialize environment
        self.env = Environment(
            G=graph,
            nodes_exit=nodes_exit,
            nodes_fire=nodes_fire,
            device=self.device,
            beta=self.hyperparameters.get("beta"),
            num_virtual_agents=self.hyperparameters.get("num_virtual_agents"),
            reward_exit=self.hyperparameters.get("reward_exit"),
            reward_fire=self.hyperparameters.get("reward_fire"),
            reward_invalid=self.hyperparameters.get("reward_invalid"),
            reward_valid=self.hyperparameters.get("reward_valid"),
            reward_congestion=self.hyperparameters.get("reward_congestion"),
        )

        # Initialize the neural network
        self.net = Network(obs_size=self.env.obs_size, n_actions=self.env.n_actions)
        if self.verbose:
            logger.info(f"Network: {self.net}")

        # Set optimizer for training
        self.optimizer = T.optim.Adam(
            self.net.parameters(), lr=self.hyperparameters.get("lr")
        )

        # Load pre-trained weights if transfer learning is enabled
        if model_pt:
            logger.info("Loading weights for transfer learning")
            if isinstance(model_pt, bytes):
                buffer = io.BytesIO(model_pt)
                state_model = T.load(buffer)
            else:
                state_model = model_pt

            # Agora carregando o modelo e otimizador:
            self.net.load_state_dict(state_model["model"])
            self.optimizer.load_state_dict(state_model["optimizer"])

        else:
            logger.info("No transfer learning applied")

        # Initialize target network (used for stabilizing training)
        self.target = Network(self.env.obs_size, self.env.n_actions)
        self.target.load_state_dict(self.net.state_dict())

        # Reward tracking
        self.reward_buffer = [0]
        self.episode_reward = 0.0

        # Experience replay buffer
        self.target_update_frequency = (
            1000  # Defines how often the target network is updated
        )
        self.action_list = np.arange(0, len(self.env.graph.nodes)).tolist()
        self.replay_buffer = deque(maxlen=self.hyperparameters.get("buffer_size"))

    def train(self, previous_state=None) -> None:
        start_time = time.time()
        state = self.env.reset(previous_state)

        state_vector = self.env.state_to_vector(state)

        if self.verbose:
            logger.info(
                f"Starting in {state % len(self.env.graph.nodes)} node. Fire: {self.env.nodes_fire}. Congestion: {self.env.agent_positions}"
            )

        loss_list = []
        mean_reward = []
        number_ep = []
        steps_per_episode = []
        episodes_steps = []

        explore_counts = []
        exploit_counts = []
        episodes_explore_exploit = []

        reward_variance = []

        state_dict = {"episodes": [], "explore_exploit": [], "time": []}

        path = [state % int(len(self.env.graph.nodes))]
        for i in tqdm(range(self.hyperparameters.get("episodes"))):
            epsilon = np.exp(-i / (self.hyperparameters.get("episodes") / 2))
            p = random.random()

            state_dict["episodes"].append(i)

            if p <= epsilon:
                neighbors = np.where(
                    self.env.adj_mat[state % len(self.env.graph.nodes)] > 0
                )[0]
                neighbors = [
                    v for v in neighbors if v != state % len(self.env.graph.nodes)
                ]
                action = random.choice(neighbors)
                state_dict["explore_exploit"].append("explore")

            else:
                vector_state = self.env.state_to_vector(int(state))
                tensor_state = T.tensor([vector_state])
                action = self.net.act(tensor_state)
                state_dict["explore_exploit"].append("exploit")

            new_state, reward, done = self.env.step(state, action)
            new_state_vector = self.env.state_to_vector(new_state)

            path.append(new_state % int(len(self.env.graph.nodes)))

            # Experience Replay
            transition = (state_vector, action, reward, done, new_state_vector)
            self.replay_buffer.append(transition)
            state = new_state
            state_vector = self.env.state_to_vector(state)
            self.episode_reward += reward

            if done:
                steps_in_episode = len(path) - 1
                steps_per_episode.append(steps_in_episode)
                episodes_steps.append(i)  # salva o episódio atual para depois plotar

                state = self.env.reset(previous_state)

                state_vector = self.env.state_to_vector(state)
                self.reward_buffer.append(self.episode_reward)

                if self.verbose:
                    counts = Counter(self.env.agent_positions)
                    congestion_nodes = sorted(
                        [
                            node
                            for node, count in counts.items()
                            if count > self.env.congestion_threshold
                        ]
                    )
                    logger.info(
                        f"Train {i} - Len: {len(path)},  Path: {path} - Reward: {self.episode_reward}"
                    )
                    logger.info(
                        f"Starting in {state % len(self.env.graph.nodes)} node. Fire: {self.env.nodes_fire}. Congestion: {congestion_nodes} - visited: {self.env.state_visit_count[state % len(self.env.graph.nodes)]}"
                    )
                path = [state % int(len(self.env.graph.nodes))]
                self.episode_reward = 0.0

            if len(self.replay_buffer) < self.hyperparameters.get("batch_size"):
                transitions = self.replay_buffer
            else:
                transitions = random.sample(
                    self.replay_buffer, self.hyperparameters.get("batch_size")
                )

            states = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rewards = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            new_states = np.asarray([t[4] for t in transitions])

            states_tensor = T.as_tensor(states, dtype=T.float32).to(self.device)
            actions_tensor = (
                T.as_tensor(actions, dtype=T.int64).to(self.device).unsqueeze(-1)
            )
            rewards_tensor = T.as_tensor(rewards, dtype=T.float32).to(self.device)
            dones_tensor = T.as_tensor(dones, dtype=T.float32).to(self.device)
            new_states_tensor = T.as_tensor(new_states, dtype=T.float32).to(self.device)

            # Target
            list_new_states_tensor = new_states_tensor
            target_q_values = self.target(list_new_states_tensor)
            max_target_q_values = target_q_values.max(dim=1, keepdim=False)[0]
            targets = (
                rewards_tensor
                + self.hyperparameters.get("gamma")
                * (1 - dones_tensor)
                * max_target_q_values
            )
            targets = targets.unsqueeze(-1)

            list_states_tensor = states_tensor

            q_values = self.net(list_states_tensor)
            action_q_values = T.gather(input=q_values, dim=1, index=actions_tensor)

            # Loss MSE
            loss = nn.functional.mse_loss(action_q_values, targets)
            loss_list.append(loss.item())
            # writer.add_scalar("Loss", loss.item(), i)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_reward.append(np.mean(self.reward_buffer))
            number_ep.append(i)

            reward_variance.append(np.var(self.reward_buffer))

            if i % self.target_update_frequency == 0:
                self.target.load_state_dict(self.net.state_dict())

            if self.verbose and i % 1000 == 0:
                logger.info("step", i, "avg rew", round(np.mean(self.reward_buffer), 2))

                n_explore = state_dict["explore_exploit"].count("explore")
                n_exploit = state_dict["explore_exploit"].count("exploit")
                explore_counts.append(n_explore / (n_explore + n_exploit))
                exploit_counts.append(n_exploit / (n_explore + n_exploit))
                episodes_explore_exploit.append(i)

        xpoints = np.array(number_ep)

        # MEAN REWARD
        ypoints = np.array(mean_reward)
        plt.ylabel("Recompensa")
        plt.xlabel("Número de Iterações")
        plt.title("Gráfico Recompensa Acumulada")
        plt.plot(xpoints, ypoints)
        plt.show()
        plt.savefig("mean.jpg")
        plt.clf()

        # VARIANCE
        ypoints = np.array(reward_variance)
        plt.ylabel("Recompensa")
        plt.xlabel("Número de Iterações")
        plt.title("Variância da Recompensa")
        plt.plot(xpoints, ypoints)
        plt.show()
        plt.savefig("variance.jpg")
        plt.clf()

        # LOSS
        plt.ylabel("Perda")
        plt.xlabel("Número de Iterações")
        plt.title("Gráfico Perda")
        ypoints = np.array(loss_list)
        plt.plot(xpoints, ypoints)
        plt.show()
        plt.savefig("loss.jpg")
        plt.clf()

        # STEPS
        def moving_average(data, window_size=100):
            return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

        # Suavizar seus dados
        smoothed_steps = moving_average(steps_per_episode, window_size=100)

        # Plot
        plt.plot(
            range(len(smoothed_steps)), smoothed_steps, label="Passos (média móvel)"
        )
        plt.xlabel("Número de Episódios")
        plt.ylabel("Quantidade de Passos")
        plt.title("Quantidade de Passos por Episódio (Suavizado)")
        plt.legend()
        plt.grid(True)
        plt.savefig("steps_per_episodes.jpg")
        plt.clf()

        # Exploração x Explotação
        plt.figure(figsize=(8, 6))
        plt.plot(
            episodes_explore_exploit,
            explore_counts,
            label="Taxa de Exploração",
            color="blue",
        )
        plt.plot(
            episodes_explore_exploit,
            exploit_counts,
            label="Taxa de Explotação",
            color="green",
        )
        plt.xlabel("Número de Iterações")
        plt.ylabel("Proporção")
        plt.title("Exploração vs Explotação")
        plt.legend()
        plt.grid(True)
        plt.savefig("explore_exploit.jpg")
        plt.show()

        state = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # if self.save_model_path:
        #     T.save(state, self.save_model_path)

        total_training_time = time.time() - start_time

        logs = {
            "loss": loss_list,  # Lista de perdas (loss) a cada episódio
            "reward": mean_reward,  # Média de recompensa a cada episódio
            "steps": number_ep,  # Número do episódio correspondente
            "steps_per_episode": steps_per_episode,  # Quantidade de passos em cada episódio
            "episodes_steps": episodes_steps,  # Número do episódio (para passos)
            "explore_counts": explore_counts,  # Proporção de exploração
            "exploit_counts": exploit_counts,  # Proporção de exploração
            "episodes_explore_exploit": episodes_explore_exploit,  # Número do episódio (para exploração/exploração)
            "total_training_time": total_training_time,  # Tempo total de treinamento (em segundos)
        }

        return state, logs
