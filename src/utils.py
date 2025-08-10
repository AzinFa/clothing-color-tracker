from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class ColorConfig:
    hsv_ranges = {
        "red":    [(0, 70, 50, 10, 255, 255), (170, 70, 50, 180, 255, 255)],
        "orange": [(10, 70, 50, 25, 255, 255)],
        "yellow": [(25, 70, 50, 35, 255, 255)],
        "green":  [(35, 40, 40, 85, 255, 255)],
        "cyan":   [(85, 40, 40, 95, 255, 255)],
        "blue":   [(95, 40, 40, 130, 255, 255)],
        "purple": [(130, 40, 40, 160, 255, 255)],
        "pink":   [(160, 40, 40, 170, 255, 255)],
        "white":  [(0, 0, 200, 180, 30, 255)],
        "gray":   [(0, 0, 60, 180, 30, 200)],
        "black":  [(0, 0, 0, 180, 255, 60)]
    }
    kmeans_k: int = 2
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    min_region_area: int = 300

def apply_clahe_bgr(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def in_range_hsv(hsv, rng):
    h,s,v = hsv
    h1,s1,v1,h2,s2,v2 = rng
    if h1 <= h2:
        return (h>=h1 and h<=h2) and (s>=s1 and s<=s2) and (v>=v1 and v<=v2)
    else:
        return ((h>=h1 or h<=h2) and (s>=s1 and s<=s2) and (v>=v1 and v<=v2))

def name_color(hsv, cfg: ColorConfig):
    if hsv is None: 
        return "unknown"
    h,s,v = hsv
    for name, ranges in cfg.hsv_ranges.items():
        for rng in ranges:
            if in_range_hsv((h,s,v), rng):
                return name
    if v < 60: return "black"
    if v > 200 and s < 30: return "white"
    if s < 30: return "gray"
    return "unknown"
