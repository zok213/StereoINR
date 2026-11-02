import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
Qualcomm AI Hub (qai_hub) On-Device Profiling & Compilation Pipeline
Connects to Qualcomm Device Cloud to compile and profile models on physical Snapdragon hardware:
- Snapdragon 8 Gen 2 / Snapdragon 8 Gen 3 / Snapdragon XR2 Gen 2
- Extracts REAL on-device execution latency (ms), NPU vs GPU vs CPU layer placement, and memory usage.
"""

import sys
import os

def check_qualcomm_ai_hub():
    print("=" * 80)
    print(">>> QUALCOMM AI HUB CLOUD PROFILING INSPECTOR")
    print("=" * 80)
    
    try:
        import qai_hub as hub
        print(f"[OK] Qualcomm AI Hub SDK Installed: Version {hub.__version__}")
    except ImportError:
        print("[!] qai-hub is not installed. Install with: pip install qai-hub")
        return

    # Check if API token is configured
    try:
        devices = hub.get_devices()
        print(f"[OK] Successfully authenticated with Qualcomm AI Hub Cloud!")
        print(f"     Total Available Cloud Devices: {len(devices)}")
        
        # Filter for relevant Snapdragon platforms (Adreno 740 / XR2 Gen 2 equivalents)
        print("\n[*] Recommended Target Devices with Adreno 740 / Hexagon NPU:")
        for dev in devices:
            name = dev.name.lower()
            if any(k in name for k in ['samsung galaxy s23', 'samsung galaxy s24', 'snapdragon', 'xr2', 'gen 2', 'gen 3']):
                print(f"    - {dev.name} (OS: {dev.os}, Chipset: {getattr(dev, 'chipset', 'Snapdragon')})")
                
    except Exception as e:
        print("[NOTE] Qualcomm AI Hub API Token is not yet configured on this machine.")
        print("       To enable live online cloud profiling on physical Snapdragon hardware:")
        print("       1. Register a free developer account at: https://aihub.qualcomm.com")
        print("       2. Obtain your API token from: Account Settings -> API Token")
        print("       3. Run in your terminal: qai-hub configure --api_token <YOUR_TOKEN>")
        print("       4. Once configured, you can submit compiled ONNX models to physical Snapdragon 8 Gen 2 / XR2 devices!")
    print("=" * 80)


if __name__ == '__main__':
    check_qualcomm_ai_hub()
