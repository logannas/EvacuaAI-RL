import json
import io
import torch
from datetime import datetime
import asyncio
import pickle
from src.core.minio_config import download_minio, upload_minio
from src.utils.create_graph import create_graph
from src.model.agent import Agent


class Train:
    def __init__(
        self,
        minio_client,
        mongodb,
        project_id: str,
        hyperparameters: dict,
        transfer_learning_version: str | None = None,
        version: str | None = None,
    ):
        self.project_id = project_id
        self.hyperparameters = hyperparameters
        self.transfer_learning_version = transfer_learning_version
        self.version = version

        self.minio_client = minio_client
        self.mongodb = mongodb

        # get infos from minio
        self.define_parameters()

    def get_model(self):
        model_pt = download_minio(
            self.minio_client,
            f"experiments/{self.project_id}/{self.transfer_learning_version}/model.pt",
        )

        return model_pt

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

    async def execute(self):
        self.hyperparameters["episodes"] = int(self.hyperparameters["episodes"] * 4)
        self.agent = Agent(
            graph=self.graph,
            nodes_exit=self.nodes_exit,
            hyperparameters=self.hyperparameters,
            verbose=True,
        )

        state, logs = self.agent.train()

        model_bytes = io.BytesIO()
        torch.save(state, model_bytes)

        logs_bytes = pickle.dumps(logs)
        json_data = json.dumps(self.hyperparameters)
        json_bytes = json_data.encode("utf-8")

        await asyncio.gather(
            self.to_minio(
                f"experiments/{self.project_id}/{self.version}/model.pt",
                model_bytes.getvalue(),
                content_type="application/octet-stream",
            ),
            self.to_minio(
                f"experiments/{self.project_id}/{self.version}/logs.pkl",
                logs_bytes,
                content_type="application/octet-stream",
            ),
            self.to_minio(
                f"experiments/{self.project_id}/{self.version}/config.json",
                json_bytes,
                content_type="application/json",
            ),
        )

        # Upload to mongodb
        self.mongodb[2].update_one(
            {"project_id": self.project_id},
            {
                "$set": {
                    "experiment_s3_uri": f"experiments/{self.project_id}",
                    "updateTime": datetime.now().isoformat(),
                }
            },
            upsert=True,  # Se quiser criar se não existir
        )

    async def to_minio(self, path: str, data: bytes, content_type: str):
        if content_type == "application/octet-stream":
            buffer = io.BytesIO(data)
            buffer.seek(0)
        else:
            buffer = data
        await upload_minio(self.minio_client, path, buffer, content_type)
