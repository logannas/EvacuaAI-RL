import grpc
import src.grpc_server.evacuai_rl_pb2 as evacuai_rl_pb2
import src.grpc_server.evacuai_rl_pb2_grpc as evacuai_rl_pb2_grpc


def run_train_model(project_id, hyp):
    # Crie o canal gRPC e o stub
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = evacuai_rl_pb2_grpc.ReinforcementLearningStub(channel)

        # Crie a requisição para o treinamento
        request = evacuai_rl_pb2.TrainRequest(
            hyperparameters=evacuai_rl_pb2.HypParams(
                beta=hyp["beta"],
                lr=hyp["lr"],
                batch_size=hyp["batchsize"],
                buffer_size=hyp["buffer_size"],
                episodes=hyp["episodes"],
                gamma=hyp["gamma"],
                epsilon=hyp["epsilon"],
                congestion_threshold=hyp["congestion_threshold"],
                num_virtual_agents=hyp["num_virtual_agents"],
            ),
            project_id=project_id,
            # transfer_learning_version = transfer_learning_version
        )

        # Faça a chamada gRPC
        response = stub.TrainModel(request)

        # Retorne o model_id retornado pela resposta
        return response.model_id


if __name__ == "__main__":
    # Exemplo de chamada para o método TrainModel
    project_id_sg11_01 = "d6cd284a5d41638e2f2669f7f190b6a8"
    project_id_sg11_02 = "5f53235fadca03762f0fd6080574dda9"
    project_id_sg11_completo = "990c6a85aef3adffe85242f3e6f4d7e1"
    hyp = {
        "beta": 1,
        "lr": 1e-2,
        "batchsize": 240,
        "buffer_size": 4000,
        "episodes": 30000,
        "gamma": 0.9,
        "epsilon": 0.7,
        "num_virtual_agents": 20,
        "congestion_threshold": 5,
    }

    model_id = run_train_model(project_id_sg11_01, hyp)
    print(f"Model trained with ID: {model_id}")
