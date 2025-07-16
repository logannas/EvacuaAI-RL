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
        request = evacuai_rl_pb2.InferenceRequest(
            project_id=project_id,
            version=version,
            previous_state=state,
            fire_nodes=fire_nodes,
            agents_positions=congestion_nodes,
        )

        # Faça a chamada gRPC
        response = stub.Inference(request)

        # Retorne as previsões retornadas pela resposta
        return response.predictions


if __name__ == "__main__":
    # Exemplo de chamada para o método TrainModel
    project_id_sg11_01 = "d6cd284a5d41638e2f2669f7f190b6a8"
    project_id_sg11_02 = "5f53235fadca03762f0fd6080574dda9"
    project_id_sg11_completo = "990c6a85aef3adffe85242f3e6f4d7e1"

    return_iference = run_inference(
        project_id_sg11_01,
        version="20250715132149",
        state=None,
        fire_nodes=[],
        congestion_nodes=[],
    )
    print(f"Return: {return_iference}")
