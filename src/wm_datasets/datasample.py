
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoSample:
    video_path: str
    label: Optional[int] = None