import ffmpeg
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

class Summarizer(object):
    def __init__(self, data):
        audio_path = self.extract_audio_from_bytes(data)
        transcript, full_text = self.transcribe_audio(audio_path)
        summary_sentences = self.summarize_text(full_text)
        timestamps = self.extract_key_timestamps(summary_sentences, transcript)
        title = "Краткий конспект видео"
        self.export_to_pdf(title, timestamps, summary_sentences)
    
    def extract_audio_from_bytes(video_bytes, output_audio_path: str = 'audio.wav'):
        process = (
            ffmpeg
            .input('pipe:', format='mp4')  # adjust if needed (e.g., 'webm', etc.)
            .output(output_audio_path, ac=1, ar='16000')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        process.stdin.write(video_bytes)
        process.stdin.close()
        process.wait()
        return output_audio_path
    
    def transcribe_audio(self, audio_path: str, model_size: str = "medium", lang: str = "ru") -> tuple[list, str]:
        # print("🧠 Распознавание речи с помощью faster-whisper...")
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        segments, _ = model.transcribe(audio_path, language=lang)

        transcript = []
        for segment in segments:
            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        full_text = " ".join([s["text"] for s in transcript])
        return transcript, full_text.strip()
    
    def deduplicate_sentences(self, sentences):
        seen = set()
        unique = []
        for sentence in sentences:
            s = sentence.strip()
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique
    
    def split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s.strip()) > 0]
    
    def summarize_text(self, text, model_name: str = "ai-forever/sbert_large_nlu_ru", summary_size: int = 7):
        # print("📚 Создание краткого конспекта (SBERT)...")
        model = SentenceTransformer(model_name)

        sentences = self.split_into_sentences(text)
        sentences = self.deduplicate_sentences(sentences)

        if len(sentences) <= summary_size:
            return sentences

        embeddings = model.encode(sentences, convert_to_tensor=True)
        centroid = embeddings.mean(dim=0)
        similarities = cosine_similarity(centroid.unsqueeze(0).cpu().numpy(), embeddings.cpu().numpy())[0]

        top_indices = similarities.argsort()[-summary_size:][::-1]
        summary = [sentences[i] for i in sorted(top_indices)]
        return summary
    
    def extract_key_timestamps(self, summary_sentences, transcript) -> list[str]:
        timestamps = []
        for s in summary_sentences:
            for entry in transcript:
                if s[:20] in entry['text']:
                    timestamps.append(f"- {entry['start']:.0f}s: {entry['text'][:60]}...")
                    break
        return timestamps
    
    def wrap_text(text, width, canvas_obj, font_name, font_size):
        words = text.split()
        lines = []
        line = ''
        space_width = canvas_obj.stringWidth(' ', font_name, font_size)
        for word in words:
            word_width = canvas_obj.stringWidth(word, font_name, font_size)
            line_width = canvas_obj.stringWidth(line, font_name, font_size)
            if line_width + word_width + space_width <= width:
                line += word + ' '
            else:
                lines.append(line.strip())
                line = word + ' '
        if line:
            lines.append(line.strip())
        return lines
    
    def export_to_pdf(self, title: str, timestamps, summary_lines, output_path="summary.pdf"):
        # print("🖨️ Экспорт в PDF...")
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        x, y = 50, height - 50

        c.setFont("DejaVuSans", 16)
        c.drawString(x, y, title)
        y -= 30

        c.setFont("DejaVuSans", 12)
        for t in timestamps:
            c.drawString(x, y, t)
            y -= 20
        y -= 10

        for s in summary_lines:
            wrapped_lines = self.wrap_text(f"• {s.strip()}", width - 2 * x, c, "DejaVuSans", 12)
            for line in wrapped_lines:
                c.drawString(x, y, line)
                y -= 18
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("DejaVuSans", 12)

        c.save()