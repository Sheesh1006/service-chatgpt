from typing import Iterator
from chatgpt_service import chatgpt_service_pb2_grpc
from chatgpt_service.chatgpt_service_pb2 import (
    GetRawNotesRequest, GetRawNotesResponse
)
from io import BytesIO
from .summarize import Summarizer


class ChatGPTServiceServicer(chatgpt_service_pb2_grpc.ChatGPTServiceServicer):
    video = []
    presentation = []

    def GetRawNotes(
        self,
        request_iterator: Iterator[GetRawNotesRequest],
        context
    ) -> Iterator[GetRawNotesResponse]:
        video_buffer = BytesIO()
        for request in request_iterator:
            video_buffer.write(request.video)
        
        frame = Summarizer(video_buffer)
        yield GetRawNotesResponse(raw_notes="HELLO WORLD!")