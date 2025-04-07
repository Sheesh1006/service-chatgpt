import logging
from concurrent import futures
import grpc
from chatgpt_service import chatgpt_service_pb2_grpc
from server.server import ChatGPTServiceServicer


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatgpt_service_pb2_grpc.add_ChatGPTServiceServicer_to_server(
        ChatGPTServiceServicer(), server
    )
    # TODO: move to config.yml later
    server.add_insecure_port("[::]:50051") 
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()