#!/usr/bin/env python3
"""
Smart installer for Decompressed.
Detects your system and recommends the optimal installation command.
"""

import subprocess
import sys


def detect_gpu():
    """Detect if GPU is available and what type."""
    print("🔍 Detecting system configuration...")
    print()
    
    # Check for NVIDIA GPU via torch
    has_nvidia = False
    has_amd = False
    has_intel = False
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Detected: {gpu_name}")
            
            if "NVIDIA" in gpu_name.upper() or "RTX" in gpu_name.upper() or "GTX" in gpu_name.upper():
                has_nvidia = True
            elif "AMD" in gpu_name.upper() or "Radeon" in gpu_name.upper():
                has_amd = True
            elif "Intel" in gpu_name.upper() or "Arc" in gpu_name.upper():
                has_intel = True
        else:
            print("❌ No GPU detected via PyTorch")
    except ImportError:
        print("⚠️  PyTorch not installed - cannot detect GPU")
        print("   (This is OK, we'll install it if needed)")
    
    # Check for CUDA Toolkit
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ CUDA Toolkit found (NVIDIA GPU support)")
            has_nvidia = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ CUDA Toolkit not found")
    
    # Check for ROCm
    try:
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ ROCm found (AMD GPU support)")
            has_amd = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ ROCm not found")
    
    print()
    return has_nvidia, has_amd, has_intel


def recommend_install(has_nvidia, has_amd, has_intel):
    """Recommend the best installation command."""
    if has_nvidia or has_amd or has_intel:
        print("📦 RECOMMENDED INSTALLATION")
        print("=" * 70)
        print()
        print("Your system has GPU support! Install with:")
        print()
        print("  pip install decompressed[gpu]")
        print()
        print("This will install:")
        print("  • NumPy (required)")
        print("  • PyTorch (GPU framework)")
        print("  • Triton (GPU-agnostic kernels)")
        print()
        
        if has_nvidia:
            print("🚀 NVIDIA GPU detected:")
            print("   You'll get CUDA native (fastest) + Triton (fast) + C++ CPU")
        elif has_amd:
            print("🚀 AMD GPU detected:")
            print("   You'll get Triton GPU (fast) + C++ CPU")
        elif has_intel:
            print("🚀 Intel GPU detected:")
            print("   You'll get Triton GPU (experimental) + C++ CPU")
    else:
        print("📦 RECOMMENDED INSTALLATION")
        print("=" * 70)
        print()
        print("No GPU detected. Install with:")
        print()
        print("  pip install decompressed")
        print()
        print("This will install:")
        print("  • NumPy (required)")
        print("  • C++ CPU extensions (if CMake available)")
        print("  • Pure Python fallback (always works)")
        print()
        print("💡 TIP: If you install a GPU later, run:")
        print("   pip install --upgrade decompressed[gpu]")
    
    print()
    print("=" * 70)


def main():
    """Main entry point."""
    print()
    print("=" * 70)
    print("DECOMPRESSED SMART INSTALLER")
    print("=" * 70)
    print()
    
    has_nvidia, has_amd, has_intel = detect_gpu()
    recommend_install(has_nvidia, has_amd, has_intel)
    
    # Ask if user wants to proceed
    print()
    response = input("Install now? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        if has_nvidia or has_amd or has_intel:
            cmd = [sys.executable, "-m", "pip", "install", "decompressed[gpu]"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "decompressed"]
        
        print()
        print(f"Running: {' '.join(cmd)}")
        print()
        subprocess.run(cmd)
    else:
        print()
        print("Installation cancelled. Run the command above when ready!")
    
    print()


if __name__ == "__main__":
    main()
