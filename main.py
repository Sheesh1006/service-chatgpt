import logging
from concurrent import futures
import grpc
from chatgpt_service import chatgpt_service_pb2_grpc
from yaml import safe_load
from munch import munchify
from server.server import ChatGPTServiceServicer


def serve() -> None:
    with open('config.yml') as cfg:
        config = munchify(safe_load(cfg))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatgpt_service_pb2_grpc.add_ChatGPTServiceServicer_to_server(
        ChatGPTServiceServicer(), server
    )
    server.add_insecure_port(config.grpc_server.addr) 
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()