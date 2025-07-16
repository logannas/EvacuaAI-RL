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
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = evacuai_rl_pb2_grpc.ReinforcementLearningStub(channel)

        # Crie a requisição para a inferência
        request = evacuai_rl_pb2.InferenceRequest(
            project_id=project_id,
            version=version,
            previous_state=62,
            fire_nodes = [49],
            agents_positions=[41, 41, 41, 41, 41, 41]
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

    # model_id_sg11_01 = run_inference(project_id_sg11_01, version="20250609203631")
    # model_id = run_inference(project_id_sg11_02, version="20250611203715")
    # model_id = run_inference(project_id=project_id_sg11_completo, version="20250712185817")
    model_id = run_inference(project_id_sg11_01, version="20250715132149")
    print(f"Model trained with ID: {model_id}")

    # # Exemplo de chamada para o método Inference
    # state = 0  # exemplo de estado
    # predictions = run_inference(state)
    # print(f"Inference result: {predictions}")
