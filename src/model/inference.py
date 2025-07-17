import io
import torch as T
from loguru import logger
from tqdm import tqdm
from src.model.network import Network
from src.model.environment import Environment


class Inference:
    def __init__(
        self,
        graph,
        nodes_exit,
        model: str | None = None,
        nodes_fire: list = [],
        previous_state: int = None,
        agents_positions: list = [],
    ):
        self.nodes_fire = nodes_fire
        self.agents_positions = agents_positions
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        logger.info(f"Fire: {nodes_fire}")
        logger.info(f"Agents Position: {agents_positions}")
        self.env = Environment(
            G=graph,
            nodes_exit=nodes_exit,
            nodes_fire=nodes_fire,
            device=self.device,
            num_virtual_agents=len(agents_positions),
            test=True,
        )

        buffer = io.BytesIO(model)
        device = T.device("cuda" if T.cuda.is_available() else "cpu")
        state_model = T.load(buffer, map_location=device)

        self.model = Network(self.env.obs_size, self.env.n_actions)
        self.model.load_state_dict(state_model["model"])
        self.model.eval()

        if previous_state:
            dict_nodes = {previous_state: previous_state}
        else:
            dict_nodes = self.env.dict_node.copy()
            for node in self.env.nodes_exit:
                dict_nodes.pop(node, None)

        self.use_virtual_agents = True if self.agents_positions else False

        self.dict_nodes = {k: [v] for k, v in dict_nodes.items()}

    def __call__(self) -> dict:
        for node in self.dict_nodes:
            state = self.env.reset(
                int(node),
                use_virtual_agents=True,
                agents_virtuais_positions=self.agents_positions,
            )
            path = [state % len(self.env.graph.nodes)]
            distance = 0
            reward_total = 0
            for i in tqdm(range(self.env.num_nodes)):
                vector_state = self.env.state_to_vector(state)
                tensor_state = T.tensor([vector_state])

                action = self.model.act(tensor_state)
                distance += self.env.adj_mat.item(state, action)
                new_state, reward, done = self.env.step(state, action)
                path.append(int(new_state % len(self.env.graph.nodes)))

                reward_total += reward

                state = new_state

                if done:
                    self.dict_nodes[node] = path
                    logger.info(
                        f"Inference {i} - Node: {node} - Path: {self.dict_nodes[node]} - Reward: {reward_total} - Distance: {distance}"
                    )
                    break

        result = []

        for _, value in self.dict_nodes.items():
            init_node = value[0] if isinstance(value, list) and value else value
            last_node = value[-1] if isinstance(value, list) and value else value
            path = value if isinstance(value, list) else [value]
            print(path)

            if self.nodes_fire and any(node in path for node in self.nodes_fire):
                path = []

            result.append(
                {
                    "init_node": int(init_node),
                    "last_node": int(last_node),
                    "path": path,
                }
            )

        return result
