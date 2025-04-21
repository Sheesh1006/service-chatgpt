from typing import Iterator
from chatgpt_service import chatgpt_service_pb2_grpc
from chatgpt_service.chatgpt_service_pb2 import (
    GetRawNotesRequest, GetRawNotesResponse,
    GetTimestampsRequest, GetTimestampsResponse,
    GetKeyFramesRequest, GetKeyFramesResponse
)
import grpc
from .summarize import Summarizer


class ChatGPTServiceServicer(chatgpt_service_pb2_grpc.ChatGPTServiceServicer):
    def GetRawNotes(
        self,
        request_iterator: Iterator[GetRawNotesRequest],
        context
    ) -> Iterator[GetRawNotesResponse]:
        video_chunks = [request.video for request in request_iterator]
        if not video_chunks:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("No video data received.")
            return
        video_bytes = b''.join(video_chunks)
        self.summarizer = Summarizer(video_bytes)
        raw_summary = self.summarizer.summarize_text()
        raw_summary = '###'.join(raw_summary)
        # chunk_size = 64 * 1024
        # for i in range(0, len(raw_summary), chunk_size):
        #     piece = raw_summary[i : i + chunk_size]
        yield GetRawNotesResponse(raw_notes=raw_summary)
    
    def GetTimestamps(
        self,
        request: GetTimestampsRequest,
        context
    ) -> Iterator[GetTimestampsResponse]:
        raw_timestamps = self.summarizer.extract_timestamps()
        yield GetTimestampsResponse(timestamps='###'.join(raw_timestamps))
    
    def GetKeyFrames(
        self,
        request: GetKeyFramesRequest,
        context
    ) -> Iterator[GetKeyFramesResponse]:
        raw_keyframes = self.summarizer.extract_keyframes()
        yield GetKeyFramesResponse(keyframes='')