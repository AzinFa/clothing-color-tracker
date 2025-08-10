from dataclasses import dataclass

@dataclass
class Cfg:
    input_video: str = "sample_videos/input_sample.mp4"
    output_video: str = "sample_videos/output_sample.mp4"
    mode: str = "precise"
    use_clahe: bool = True
    kmeans_k: int = 2
    tracker_iou_th: float = 0.5
    tracker_max_age: int = 10
    hsv_momentum: float = 0.7
    min_region_area: int = 300
    show_debug: bool = False

CFG = Cfg()
print(CFG)
