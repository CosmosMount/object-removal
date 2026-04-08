from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    dynamic_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    motion_threshold: float = 1.5
    dilation_kernel: int = 15
    adaptive_dilation: bool = True
    lk_max_corners: int = 60
    temp_bg_window: int = 40
    inpaint_mode: str = "both"
    max_frames: Optional[int] = None
    output_fps: Optional[float] = None
    mog2_history: int = 200
    mog2_threshold: int = 40
    min_blob_area: int = 800
    bg_color: str = "#16162a"
