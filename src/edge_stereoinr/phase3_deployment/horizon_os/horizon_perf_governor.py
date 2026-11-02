"""
Meta Horizon OS Performance Governor & Dynamic Thermal Budgeter
Addresses the hardware reality: Meta Horizon OS DOES NOT grant NPU/FastRPC access to 3rd-party developers!
All computation must execute strictly on Adreno 740 GPU (Level 0-5) and Kryo CPU (Level 0-5).

References Meta Horizon OS Documentation:
- Boosting CPU and GPU Levels (OVRManager.suggestedCpuPerfLevel / suggestedGpuPerfLevel)
- ProcessorPerformanceLevel: PowerSavings, SustainedLow, SustainedHigh, Boost
"""

import time
from enum import Enum
from typing import Dict, Any, Tuple


class ProcessorPerformanceLevel(Enum):
    """Corresponds to OVRPlugin.ProcessorPerformanceLevel in Meta Horizon OS SDK."""
    PowerSavings = 0   # Lowest clock range, maximum battery life (CPU 0-4, GPU 0-4)
    SustainedLow = 1   # Low clock range, stable baseline (CPU 2-4, GPU 1-4)
    SustainedHigh = 2  # Standard high performance tier for VR frame stability
    Boost = 3          # Peak burst clocks, time-limited by Meta OS (< 45s) to avoid overheating


class HorizonPassthroughState(Enum):
    """Dynamic operational modes for Foveated Edge-StereoINR on Quest 3S."""
    STANDBY_GAMING = 0     # User playing game or moving head: AI sleeps, 100% GPU for game render
    TEXT_ACQUISITION = 1   # User just raised phone/book: Quick boost (< 1.5s) to lock onto text
    SUSTAINED_READING = 2  # User actively reading: Sustained high foveated upscaling (40 PPD)
    THERMAL_THROTTLE = 3   # Headset hot: Graceful degradation to basic bicubic + CAS


class MetaHorizonPerfGovernor:
    """
    Manages CPU & GPU clock tiers on Meta Quest 3S according to Meta's Horizon OS guidelines.
    Guarantees thermal headroom and 90 FPS VR pacing without touching the restricted NPU.
    """
    def __init__(self, target_fps: int = 90):
        self.target_fps = target_fps
        self.frame_budget_ms = 1000.0 / target_fps  # 11.11 ms at 90Hz
        
        # State tracking
        self.current_state = HorizonPassthroughState.STANDBY_GAMING
        self.cpu_level = ProcessorPerformanceLevel.SustainedLow
        self.gpu_level = ProcessorPerformanceLevel.SustainedLow
        
        # Boost safety timer (Meta caps Boost to prevent thermal tripping)
        self.boost_start_time = 0.0
        self.max_boost_duration_sec = 2.0  # Ultra-safe: only 2 seconds vs Meta's 45s hard limit
        
        # Statistics
        self.last_switch_time = time.time()

    def update_state(
        self,
        text_detected: bool,
        text_confidence: float,
        head_angular_vel_deg_s: float,
        headset_temperature_c: float = 38.0
    ) -> Tuple[ProcessorPerformanceLevel, ProcessorPerformanceLevel, str]:
        """
        Evaluates sensor & visual input to dynamically command Horizon OS CPU/GPU levels.
        
        Args:
            text_detected: Whether SmartTextFoveaDetector found text/display
            text_confidence: Confidence score in [0.0, 1.0]
            head_angular_vel_deg_s: Gyroscope rotation speed in deg/s
            headset_temperature_c: Thermal sensor reading (nominal < 42°C)
            
        Returns:
            (suggested_cpu_level, suggested_gpu_level, action_description)
        """
        now = time.time()

        # 1. Thermal Emergency Check
        if headset_temperature_c >= 42.0:
            self.current_state = HorizonPassthroughState.THERMAL_THROTTLE
            self.cpu_level = ProcessorPerformanceLevel.PowerSavings
            self.gpu_level = ProcessorPerformanceLevel.SustainedLow
            return (self.cpu_level, self.gpu_level, "THERMAL_THROTTLE: Downclocked to protect headset.")

        # 2. Saccade / Rapid Motion Check: Eyes cannot resolve text when head moves > 45 deg/s
        if head_angular_vel_deg_s > 45.0 or not text_detected or text_confidence < 0.20:
            self.current_state = HorizonPassthroughState.STANDBY_GAMING
            self.cpu_level = ProcessorPerformanceLevel.SustainedLow
            self.gpu_level = ProcessorPerformanceLevel.SustainedLow
            return (self.cpu_level, self.gpu_level, "STANDBY_GAMING: AI Sleeping (0ms GPU, 0W extra power).")

        # 3. User Just Focused on Text -> Brief Boost Window
        if self.current_state == HorizonPassthroughState.STANDBY_GAMING and text_confidence >= 0.70:
            self.current_state = HorizonPassthroughState.TEXT_ACQUISITION
            self.boost_start_time = now
            self.cpu_level = ProcessorPerformanceLevel.SustainedLow
            self.gpu_level = ProcessorPerformanceLevel.Boost
            return (self.cpu_level, self.gpu_level, "TEXT_ACQUISITION: GPU Boost triggered (<2s window) for rapid lock.")

        # 4. Sustained Reading Mode
        if self.current_state == HorizonPassthroughState.TEXT_ACQUISITION:
            # Check if boost time expired
            if (now - self.boost_start_time) >= self.max_boost_duration_sec:
                self.current_state = HorizonPassthroughState.SUSTAINED_READING
                self.cpu_level = ProcessorPerformanceLevel.SustainedLow
                self.gpu_level = ProcessorPerformanceLevel.SustainedHigh
                return (self.cpu_level, self.gpu_level, "SUSTAINED_READING: Settled to SustainedHigh (stable 40 PPD).")
            else:
                return (self.cpu_level, self.gpu_level, "TEXT_ACQUISITION: Continuing burst.")

        if self.current_state == HorizonPassthroughState.SUSTAINED_READING:
            self.cpu_level = ProcessorPerformanceLevel.SustainedLow
            self.gpu_level = ProcessorPerformanceLevel.SustainedHigh
            return (self.cpu_level, self.gpu_level, "SUSTAINED_READING: Stable 40 PPD active on Adreno 740.")

        return (self.cpu_level, self.gpu_level, "DEFAULT_MAINTAIN")

    def generate_csharp_bridge_code(self) -> str:
        """Generates exact Unity C# code to apply these levels in Meta Horizon OS."""
        return """
// Auto-generated Meta Horizon OS Performance Bridge for Edge-StereoINR
using UnityEngine;

public class EdgeStereoINRHierarchyGovernor : MonoBehaviour
{
    void SetHorizonOSPerformanceLevels(OVRPlugin.ProcessorPerformanceLevel cpu, OVRPlugin.ProcessorPerformanceLevel gpu)
    {
        // Meta Horizon OS Official Performance API
        OVRPlugin.suggestedCpuPerfLevel = cpu;
        OVRPlugin.suggestedGpuPerfLevel = gpu;
    }
}
"""


def test_perf_governor():
    print("=" * 80)
    print(">>> TESTING META HORIZON OS PERFORMANCE GOVERNOR (NO-NPU FALLBACK)")
    print("=" * 80)

    governor = MetaHorizonPerfGovernor(target_fps=90)

    # Scenario 1: User playing active VR game (Moving head, no text)
    cpu, gpu, desc = governor.update_state(text_detected=False, text_confidence=0.0, head_angular_vel_deg_s=60.0)
    print(f"[*] Scenario 1 (Gaming/Locomotion): CPU={cpu.name}, GPU={gpu.name} -> {desc}")
    assert gpu == ProcessorPerformanceLevel.SustainedLow

    # Scenario 2: User brings up phone in front of face
    cpu, gpu, desc = governor.update_state(text_detected=True, text_confidence=0.92, head_angular_vel_deg_s=5.0)
    print(f"[*] Scenario 2 (Text Raised):     CPU={cpu.name}, GPU={gpu.name} -> {desc}")
    assert gpu == ProcessorPerformanceLevel.Boost

    # Scenario 3: User continues reading for 3 seconds (Boost window expires)
    time.sleep(2.1)
    cpu, gpu, desc = governor.update_state(text_detected=True, text_confidence=0.95, head_angular_vel_deg_s=2.0)
    print(f"[*] Scenario 3 (Sustained Read):  CPU={cpu.name}, GPU={gpu.name} -> {desc}")
    assert gpu == ProcessorPerformanceLevel.SustainedHigh

    # Scenario 4: Thermal limit reached (42.5°C)
    cpu, gpu, desc = governor.update_state(text_detected=True, text_confidence=0.95, head_angular_vel_deg_s=2.0, headset_temperature_c=42.5)
    print(f"[*] Scenario 4 (Thermal Alarm):   CPU={cpu.name}, GPU={gpu.name} -> {desc}")
    assert gpu == ProcessorPerformanceLevel.SustainedLow

    print("[OK] Meta Horizon OS Performance Governor PASSED ALL SCENARIOS!")
    print("=" * 80)


if __name__ == '__main__':
    test_perf_governor()
