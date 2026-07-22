from youtube_transcript_api import YouTubeTranscriptApi

from src.services.processing.scene_extraction.interface import (
    ExtractionResult
)
from src.services.processing.scene_extraction.utils import (
    get_current_time,
    extract_video_id_from_url
)


class TranscriptExtractor:
    """YouTube transcript extractor implementation."""
    
    def extract_transcript(self, url: str) -> ExtractionResult:
        
        try:
            video_id = extract_video_id_from_url(url)
            ytt_api = YouTubeTranscriptApi()
            raw_segments = ytt_api.fetch(video_id, languages=["en", "vi"])
            
            segments = [
                {
                    "start": seg.start,
                    "end": round(seg.start + seg.duration, 2),
                    "transcript": seg.text
                }
                for seg in raw_segments
            ]
            
            
            return ExtractionResult(
                segments=segments,
                created_at=get_current_time()
            )
            
        except Exception as e:
            (f"Failed to extract transcript: {str(e)}")