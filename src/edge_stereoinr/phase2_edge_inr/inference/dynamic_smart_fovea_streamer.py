"""
Dynamic Smart-Foveated Stereoscopic Streaming Engine for Meta Quest 3S:
Integrates the sub-millisecond SmartTextFoveaDetector ("AI Reading Glasses")
with the Continuous FoveatedStereoStreamer.

Solves the real-world XR Mixed Reality problem:
- In VR/MR Passthrough, users look at smartphones, books, and monitors at off-axis angles.
- Because Quest 3S lacks physical eye-tracking cameras, fixed optical foveation at (0, 0)
  causes off-axis text/screens to fall into the blurred peripheral zone.
- DynamicSmartFoveaStreamer automatically steers the 4.0x super-resolution sweet spot
  directly to the salience centroid of the text/display in < 0.15 ms!
"""

import math
import time
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.edge_stereoinr.phase2_edge_inr.foveation.smart_text_fovea import SmartTextFoveaDetector
from src.edge_stereoinr.phase2_edge_inr.inference.foveated_streamer import FoveatedStereoStreamer


class DynamicSmartFoveaStreamer:
    """
    Sub-millisecond Salience-Guided Dynamic Foveated Streaming Engine.
    Combines text/display region proposals with continuous arbitrary-scale INR.
    """
    def __init__(
        self,
        model: nn.Module,
        tile_size: int = 64,
        overlap: int = 12,
        fovea_radius_inner: float = 0.35,
        fovea_radius_outer: float = 0.65,
        fovea_scale: float = 4.0,
        periphery_scale: float = 2.0,
        smooth_alpha: float = 0.35,  # EMA temporal smoothing for gaze center
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(model.parameters()).device
        self.fovea_detector = SmartTextFoveaDetector().to(self.device).eval()
        self.streamer = FoveatedStereoStreamer(
            model=model,
            tile_size=tile_size,
            overlap=overlap,
            fovea_radius_inner=fovea_radius_inner,
            fovea_radius_outer=fovea_radius_outer,
            fovea_scale=fovea_scale,
            periphery_scale=periphery_scale,
            device=self.device,
        )
        self.smooth_alpha = smooth_alpha
        self.current_center = torch.zeros(2, device=self.device)  # (cx, cy)
        
    def stream_frame(
        self,
        img_l: torch.Tensor,
        img_r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Processes stereo frame with auto-detected dynamic foveation:
        1. Sub-millisecond gradient salience analysis to locate text/display centroid.
        2. Temporal EMA smoothing to prevent saccadic jitter.
        3. Foveated super-resolution centered on the target object.
        """
        t0 = time.perf_counter()
        
        # 1. Detect text / high-frequency salience
        with torch.no_grad():
            detected_center, salience_map, confidence = self.fovea_detector(img_l)
            
        # 2. Smooth center update
        if confidence > 0.25:
            target_cx = float(detected_center[0, 0].item())
            target_cy = float(detected_center[0, 1].item())
        else:
            # Fall back to optical axis
            target_cx = 0.0
            target_cy = 0.0
            
        cur_cx = float(self.current_center[0].item())
        cur_cy = float(self.current_center[1].item())
        
        smooth_cx = (1.0 - self.smooth_alpha) * cur_cx + self.smooth_alpha * target_cx
        smooth_cy = (1.0 - self.smooth_alpha) * cur_cy + self.smooth_alpha * target_cy
        
        self.current_center[0] = smooth_cx
        self.current_center[1] = smooth_cy
        
        t_detect = (time.perf_counter() - t0) * 1000.0
        
        # 3. Execute foveated super-resolution
        sr_l, sr_r, fov_stats = self.streamer.stream_foveated_super_resolve(
            img_l, img_r, center=(smooth_cx, smooth_cy)
        )
        
        total_latency = fov_stats["latency_ms"] + t_detect
        
        stats = {
            "gaze_center": (smooth_cx, smooth_cy),
            "confidence": float(confidence),
            "detection_latency_ms": t_detect,
            "streaming_latency_ms": fov_stats["latency_ms"],
            "total_latency_ms": total_latency,
            "fovea_tiles": fov_stats["fovea_tiles"],
            "periph_tiles": fov_stats["periph_tiles"],
            "fps": 1000.0 / max(total_latency, 1e-4),
        }
        
        return sr_l, sr_r, stats
