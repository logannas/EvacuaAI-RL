import grpc
import asyncio
from concurrent import futures
import src.grpc_server.evacuai_rl_pb2 as evacuai_rl_pb2
import src.grpc_server.evacuai_rl_pb2_grpc as evacuai_rl_pb2_grpc
import threading
from src.core.minio_config import get_minio_client
from src.db.mongodb import get_mongodb
import logging
from src.model.train import Train
from src.model.test import Test
from datetime import datetime

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphDeepRLServicer:
    def __init__(self):
        self.agent = None
        self.minio_client = get_minio_client()
        self.mongodb = get_mongodb()

    import threading

    def TrainModel(self, request, context):
        project_id = request.project_id
        transfer_learning_version = (
            request.transfer_learning_version
            if request.HasField("transfer_learning_version")
            else None
        )

        hyperparameters = {
            "beta": request.hyperparameters.beta
            if request.hyperparameters.HasField("beta")
            else 1.0,
            "lr": request.hyperparameters.lr
            if request.hyperparameters.HasField("lr")
            else 0.001,
            "batch_size": request.hyperparameters.batch_size
            if request.hyperparameters.HasField("batch_size")
            else 32,
            "buffer_size": request.hyperparameters.buffer_size
            if request.hyperparameters.HasField("buffer_size")
            else 100000,
            "episodes": request.hyperparameters.episodes
            if request.hyperparameters.HasField("episodes")
            else 10000,
            "gamma": request.hyperparameters.gamma
            if request.hyperparameters.HasField("gamma")
            else 0.99,
            "epsilon": request.hyperparameters.epsilon
            if request.hyperparameters.HasField("epsilon")
            else 0.1,
            "congestion_threshold": request.hyperparameters.congestion_threshold
            if request.hyperparameters.HasField("congestion_threshold")
            else 5,
            "reward_exit": request.hyperparameters.reward_exit
            if request.hyperparameters.HasField("reward_exit")
            else 10000,
            "reward_fire": request.hyperparameters.reward_fire
            if request.hyperparameters.HasField("reward_fire")
            else 10000,
            "reward_invalid": request.hyperparameters.reward_invalid
            if request.hyperparameters.HasField("reward_invalid")
            else 5000,
            "reward_valid": request.hyperparameters.reward_valid
            if request.hyperparameters.HasField("reward_valid")
            else 1,
            "reward_congestion": request.hyperparameters.reward_congestion
            if request.hyperparameters.HasField("reward_congestion")
            else 3000,
        }

        self.version = datetime.now().strftime("%Y%m%d%H%M%S")

        def background_training():
            train = Train(
                self.minio_client,
                self.mongodb,
                project_id,
                hyperparameters,
                transfer_learning_version=transfer_learning_version,
                version=self.version,
            )
            asyncio.run(train.execute())

        # Inicia em background
        thread = threading.Thread(target=background_training)
        thread.start()

        return evacuai_rl_pb2.TrainResponse(model_id=self.version)

    def Inference(self, request, context):
        project_id = request.project_id
        version = request.version
        previous_state = request.previous_state if request.previous_state else None
        fire_nodes = list(request.fire_nodes) if request.fire_nodes else []
        agents_positions = (
            list(request.agents_positions) if request.agents_positions else []
        )

        teste = Test(
            minio_client=self.minio_client,
            project_id=project_id,
            version=version,
            previous_state=previous_state,
            fire_nodes=fire_nodes,
            agents_positions=agents_positions,
        )

        result = teste.execute()

        response = evacuai_rl_pb2.InferenceResponse(
            predictions=[
                evacuai_rl_pb2.Path(
                    init_node=pred["init_node"],
                    last_node=pred["last_node"],
                    path=pred["path"],
                )
                for pred in result
            ]
        )

        return response

    def GetRewardPath(self, request, context):
        reward = 0

        if request.path:
            project_id = request.project_id
            version = request.version
            path = request.path
            fire_nodes = list(request.fire_nodes) if request.fire_nodes else []
            agents_positions = (
                list(request.agents_positions) if request.agents_positions else []
            )

            getreward_path = Test(
                minio_client=self.minio_client,
                project_id=project_id,
                version=version,
                path=path,
                fire_nodes=fire_nodes,
                agents_positions=agents_positions,
            )

            reward = getreward_path.execute_reward_path()

        return evacuai_rl_pb2.GetRewardPathResponse(reward=reward)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    evacuai_rl_pb2_grpc.add_ReinforcementLearningServicer_to_server(
        GraphDeepRLServicer(), server
    )

    server.add_insecure_port("[::]:50051")
    print("Server started at port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
