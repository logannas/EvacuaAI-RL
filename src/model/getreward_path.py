import io
import torch as T
from loguru import logger
from tqdm import tqdm
from src.model.network import Network
from src.model.environment import Environment


class GetRewardPath:
    def __init__(
        self,
        graph,
        nodes_exit,
        model: str | None = None,
        nodes_fire: list = [],
        path: int = None,
        agents_positions: list = [],
    ):
        self.nodes_fire = nodes_fire
        self.agents_positions = agents_positions
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.path = path
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
        state_model = T.load(buffer)

        self.model = Network(self.env.obs_size, self.env.n_actions)
        self.model.load_state_dict(state_model["model"])
        self.model.eval()

        self.use_virtual_agents = True if self.agents_positions else False

    def __call__(self) -> dict:
        start_node = self.path[0]
        state = self.env.reset(
            int(start_node),
            use_virtual_agents=True,
            agents_virtuais_positions=self.agents_positions,
        )

        final_reward = 0

        for i in tqdm(self.path[1:]):
            action = i
            new_state, reward, done = self.env.step(state, action)
            state = new_state
            final_reward += reward

        return final_reward
