from typing import Iterator
from chatgpt_service import chatgpt_service_pb2_grpc
from chatgpt_service.chatgpt_service_pb2 import (
    GetRawNotesRequest, GetRawNotesResponse
)

class ChatGPTServiceServicer(chatgpt_service_pb2_grpc.ChatGPTServiceServicer):
    video = []
    presentation = []

    def GetRawNotes(
        self,
        request_iterator: Iterator[GetRawNotesRequest],
        context
    ) -> Iterator[GetRawNotesResponse]:
        for request in request_iterator:
            self.video.append(request.video)
            self.presentation.append(request.presentation)
        yield GetRawNotesResponse(raw_notes="HELLO WORLD!")