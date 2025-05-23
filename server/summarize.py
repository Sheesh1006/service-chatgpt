import ffmpeg
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import easyocr
from PIL import Image
import imagehash
import soundfile as sf
from io import BytesIO
import tempfile
import re


class Summarizer(object):
    def __init__(self, data):
        self.video_bytes = data
        self.reader = easyocr.Reader(['en', 'ru'])
        print("Going to work")
        audio_data = self._extract_audio(data)
        print("Extracted Audio")
        self.transcript, self.full_text = self._transcribe_audio(audio_data)
        print("Transcribed Audio")
    
    def _has_table(self, image: Image, min_lines=10):
        img_np = np.array(image.convert("RGB"))
        try:
            result = self.reader.readtext(img_np, detail=0)
            return len(result) >= min_lines
        except Exception as e:
            print(f"⚠️ Ошибка при анализе {img_np}: {e}")
            return False

    def _crop_table_region(self, image: Image):
        img_np = np.array(image.convert("RGB"))
        result = self.reader.readtext(img_np)
        if not result:
            return None

        coords = []
        for block in result:
            points = block[0]
            for x, y in points:
                coords.append((int(x), int(y)))

        if not coords:
            return None

        xs, ys = zip(*coords)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cropped = image.crop((x_min - 10, y_min - 10, x_max + 10, y_max + 10))
        return cropped

    def _split_into_sentences(self, text):
        return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s) > 0]

    def _deduplicate(self, sentences):
        seen, result = set(), []
        for s in sentences:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def summarize_text(self, model_name="ai-forever/sbert_large_nlu_ru", size=7):
        model = SentenceTransformer(model_name)
        sents = self._deduplicate(self._split_into_sentences(self.full_text))
        if len(sents) <= size:
            return sents
        emb = model.encode(sents, convert_to_tensor=True)
        center = emb.mean(dim=0)
        sims = cosine_similarity(center.unsqueeze(0).cpu().numpy(), emb.cpu().numpy())[0]
        top = sorted(np.argsort(sims)[-size:])
        self.summary = [sents[i] for i in top]
        return self.summary
    
    def _extract_audio(self, video_bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
            temp_video.write(video_bytes)
            temp_video.flush()

            try:
                out, err = (
                    ffmpeg
                    .input(temp_video.name)
                    .output('pipe:1', format='wav', ac=1, ar='16000')
                    .overwrite_output()
                    .run(
                        capture_stdout=True,
                        capture_stderr=True
                    )
                )
                return out
            except ffmpeg.Error as e:
                print("⚠️ ffmpeg error:\n", e.stderr.decode(errors="ignore"))
                raise

    def _transcribe_audio(self, audio_bytes, model_size="medium", lang="ru"):
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        audio_buffer = BytesIO(audio_bytes)
        audio_data, _ = sf.read(audio_buffer)

        segments, _ = model.transcribe(audio_data, language=lang)
        transcript = []
        for segment in segments:
            transcript.append({"start": segment.start, "text": segment.text.strip()})
        full_text = " ".join([s["text"] for s in transcript])
        return transcript, full_text.strip()

    def extract_timestamps(self):
        result = []
        for s in self.summary:
            for seg in self.transcript:
                if s[:20] in seg["text"]:
                    result.append(f"- {int(seg['start'])}s: {seg['text'][:60]}...")
                    break
        return result

    def extract_keyframes(self, interval_sec=10, max_frames=20, hash_threshold=5) -> list[Image.Image]:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(self.video_bytes)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = 0
            extracted = 0
            table_frames = []
            prev_hash = None

            while cap.isOpened() and extracted < max_frames:
                frame_idx = int(count * fps * interval_sec)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if self._has_table(pil_frame):
                    cropped = self._crop_table_region(pil_frame)
                    if cropped is not None:
                        curr_hash = imagehash.average_hash(cropped)
                        if prev_hash is None or abs(curr_hash - prev_hash) >= hash_threshold:
                            table_frames.append(cropped)
                            prev_hash = curr_hash

                extracted += 1
                count += 1

            cap.release()
        return table_frames