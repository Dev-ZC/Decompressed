#!/bin/bash
# Rebuild script to fix CUDA PTX toolchain errors
# Run this after updating CMakeLists.txt with multi-architecture support

set -e

echo "=========================================="
echo "REBUILDING DECOMPRESSED WITH CUDA FIX"
echo "=========================================="
echo ""

echo "📝 Step 1: Cleaning previous build artifacts..."
pip uninstall -y decompressed 2>/dev/null || true
rm -rf build/ _skbuild/ dist/ *.egg-info
echo "✅ Clean complete"
echo ""

echo "📦 Step 2: Rebuilding with multi-architecture CUDA support..."
echo "   This will compile for: V100, T4, A100, RTX 30xx/40xx, H100"
echo ""
pip install -e . -v
echo "✅ Rebuild complete"
echo ""

echo "🧪 Step 3: Testing CUDA backend..."
python3 << 'EOF'
try:
    from decompressed import load_cvc, pack_cvc, get_available_backends
    import numpy as np
    import torch
    
    print("\n📊 Backend Availability:")
    backends = get_available_backends()
    for name, available in backends.items():
        status = "✅" if available else "❌"
        print(f"   {status} {name.upper()}")
    
    if backends['cuda']:
        print("\n🔬 Testing CUDA with multiple loads (catches PTX errors)...")
        
        # Create test data
        vectors = np.random.randn(1000, 768).astype(np.float32)
        pack_cvc(vectors, "test_rebuild.cvc", compression="fp16")
        
        # Test with multiple iterations to ensure no PTX errors
        for i in range(3):
            result = load_cvc("test_rebuild.cvc", device="cuda", backend="cuda")
            torch.cuda.synchronize()  # Force error checking
            print(f"   ✅ Load {i+1} successful: {result.shape}")
        
        print("\n🎉 SUCCESS! CUDA backend is working correctly!")
        print("   PTX error is FIXED - multiple loads work without errors")
    else:
        print("\n⚠️  CUDA backend not available")
        print("   Check that CUDA Toolkit is installed and GPU is accessible")
        
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check CUDA/PyTorch version compatibility")
    print("2. Verify GPU is accessible: nvidia-smi")
    print("3. See CUDA_PTX_FIX.md for details")
    exit(1)
EOF

echo ""
echo "=========================================="
echo "REBUILD AND TEST COMPLETE!"
echo "=========================================="
echo ""
echo "📖 For technical details, see: CUDA_PTX_FIX.md"
