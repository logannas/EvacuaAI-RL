import json
from datetime import datetime
from loguru import logger
from src.core.minio_config import download_minio, list_files_minio
from src.utils.create_graph import create_graph
from src.model.inference import Inference
from src.model.getreward_path import GetRewardPath


class Test:
    def __init__(
        self,
        minio_client,
        project_id: str,
        version: str = None,
        previous_state: int = None,
        path: list = [],
        fire_nodes: list = [],
        agents_positions: list = [],
    ):
        self.project_id = project_id
        self.version = version
        self.previous_state = previous_state
        self.fire_nodes = fire_nodes
        self.agents_positions = agents_positions
        self.path = path

        logger.info(f"Project Id: {self.project_id}")
        logger.info(f"Version: {self.version}")

        self.minio_client = minio_client

        self.define_parameters()

    def define_parameters(self):
        graph_bytes = download_minio(self.minio_client, f"graph/{self.project_id}.json")

        string_data = graph_bytes.decode("utf-8")
        graph_dict = json.loads(string_data)["graph_filter"]

        self.graph = create_graph(
            edges=graph_dict["edges"],
            positions=graph_dict["node_coordinates"],
            nodes_exit=graph_dict["exits"],
        )

        self.nodes_exit = graph_dict["exits"]

        if not self.version:
            files = list_files_minio(
                self.minio_client, f"experiments/{self.project_id}/"
            )
            self.version = self.get_latest_timestamp(files)

        self.model_pt = download_minio(
            self.minio_client, f"experiments/{self.project_id}/{self.version}/model.pt"
        )

    def execute(self):
        test = Inference(
            graph=self.graph,
            nodes_exit=self.nodes_exit,
            nodes_fire=self.fire_nodes,
            model=self.model_pt,
            agents_positions=self.agents_positions,
            previous_state=self.previous_state,
        )

        result = test()
        return result

    def execute_reward_path(self):
        reward = GetRewardPath(
            graph=self.graph,
            nodes_exit=self.nodes_exit,
            nodes_fire=self.fire_nodes,
            model=self.model_pt,
            agents_positions=self.agents_positions,
            path=self.path,
        )

        result = reward()
        return result

    @staticmethod
    def get_latest_timestamp(paths):
        if not paths:
            return None

        timestamps = []
        for path in paths:
            try:
                ts_str = path.rstrip("/").split("/")[-1]
                ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                timestamps.append((ts, ts_str))
            except ValueError:
                continue
        if not timestamps:
            return None

        timestamps.sort(reverse=True)
        return timestamps[0][1]
