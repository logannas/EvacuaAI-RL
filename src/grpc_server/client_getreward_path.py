import grpc
import src.grpc_server.evacuai_rl_pb2 as evacuai_rl_pb2
import src.grpc_server.evacuai_rl_pb2_grpc as evacuai_rl_pb2_grpc


def run_inference(
    project_id,
    version: str = None,
    state: int = None,
    fire_nodes: list = [],
    congestion_nodes: list = [],
):
    # Crie o canal gRPC e o stub
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = evacuai_rl_pb2_grpc.ReinforcementLearningStub(channel)

        # Crie a requisição para a inferência
        request = evacuai_rl_pb2.GetRewardPathRequest(
            project_id=project_id,
            version=version,
            path=[58, 55, 50, 45, 42, 34, 32, 25, 0],
            fire_nodes=[49],
            agents_positions=[41, 41, 41, 41, 41, 41],
        )

        # Faça a chamada gRPC
        response = stub.GetRewardPath(request)

        # Retorne as previsões retornadas pela resposta
        return response.reward


if __name__ == "__main__":
    # Exemplo de chamada para o método TrainModel
    project_id = "990c6a85aef3adffe85242f3e6f4d7e1"
    project_id_1 = "d6cd284a5d41638e2f2669f7f190b6a8"

    reward = run_inference(project_id_1, version="20250715105702")
    print(f"Path Reward: {reward}")
