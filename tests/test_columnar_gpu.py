"""Tests for GPU backend support in columnar loader."""

import numpy as np
import pytest
from pathlib import Path

from decompressed import (
    pack_cvc_columns,
    load_cvc_columns,
    get_available_backends
)


# Check GPU availability
backends = get_available_backends()
HAS_CUDA = backends.get('cuda', False)
HAS_TRITON = backends.get('triton', False)
HAS_ANY_GPU = HAS_CUDA or HAS_TRITON

# Skip all tests if no GPU available
pytestmark = pytest.mark.skipif(
    not HAS_ANY_GPU,
    reason="No GPU backend available (CUDA or Triton required)"
)


class TestGPUBackendBasic:
    """Basic GPU backend tests for columnar loader."""
    
    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    def test_load_single_column_gpu_triton(self, tmp_path):
        """Test loading single column with Triton backend."""
        n = 1000
        data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path))
        
        # Load with Triton backend
        loaded = load_cvc_columns(
            str(file_path),
            columns=["text"],
            device="cuda",
            backend="triton"
        )
        
        # Verify on GPU
        import torch
        assert isinstance(loaded["text"], torch.Tensor)
        assert loaded["text"].is_cuda
        assert loaded["text"].shape == (n, 768)
        
        # Verify accuracy (FP16 tolerance)
        loaded_cpu = loaded["text"].cpu().numpy()
        np.testing.assert_allclose(loaded_cpu, data["text"], rtol=0.01, atol=0.01)
    
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_load_single_column_gpu_cuda(self, tmp_path):
        """Test loading single column with CUDA backend."""
        n = 1000
        data = {
            "embeddings": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path))
        
        # Load with CUDA backend
        loaded = load_cvc_columns(
            str(file_path),
            device="cuda",
            backend="cuda"
        )
        
        # Verify on GPU
        import torch
        assert isinstance(loaded["embeddings"], torch.Tensor)
        assert loaded["embeddings"].is_cuda
        assert loaded["embeddings"].shape == (n, 768)
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_load_multiple_columns_gpu_auto(self, tmp_path):
        """Test loading multiple columns with auto backend selection."""
        n = 500
        data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
            "audio": np.random.randn(n, 256).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path))
        
        # Load all columns with auto backend
        loaded = load_cvc_columns(
            str(file_path),
            device="cuda",
            backend="auto"
        )
        
        # Verify all on GPU
        import torch
        assert all(isinstance(v, torch.Tensor) for v in loaded.values())
        assert all(v.is_cuda for v in loaded.values())
        assert loaded["text"].shape == (n, 768)
        assert loaded["image"].shape == (n, 512)
        assert loaded["audio"].shape == (n, 256)


class TestGPUBackendCompression:
    """Test GPU backend with different compression types."""
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_fp16_compression_gpu(self, tmp_path):
        """Test FP16 compression with GPU backend."""
        n = 1000
        data = {
            "embeddings": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path), compressions={"embeddings": "fp16"})
        
        # Load on GPU
        loaded = load_cvc_columns(str(file_path), device="cuda")
        
        # Verify
        import torch
        assert loaded["embeddings"].is_cuda
        loaded_cpu = loaded["embeddings"].cpu().numpy()
        np.testing.assert_allclose(loaded_cpu, data["embeddings"], rtol=0.01, atol=0.01)
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_int8_compression_gpu(self, tmp_path):
        """Test INT8 compression with GPU backend."""
        n = 1000
        data = {
            "embeddings": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path), compressions={"embeddings": "int8"})
        
        # Load on GPU
        loaded = load_cvc_columns(str(file_path), device="cuda")
        
        # Verify (INT8 has lower accuracy)
        import torch
        assert loaded["embeddings"].is_cuda
        loaded_cpu = loaded["embeddings"].cpu().numpy()
        np.testing.assert_allclose(loaded_cpu, data["embeddings"], rtol=0.1, atol=0.1)
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_mixed_compression_gpu(self, tmp_path):
        """Test mixed compression types with GPU backend."""
        n = 500
        data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
            "doc_id": np.arange(n, dtype=np.int32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path), compressions={
            "text": "fp16",
            "image": "int8",
            "doc_id": "none"
        })
        
        # Load on GPU
        loaded = load_cvc_columns(str(file_path), device="cuda")
        
        # Verify all on GPU
        import torch
        assert all(v.is_cuda for v in loaded.values())


class TestGPUBackendSelectiveLoading:
    """Test GPU backend with selective column loading."""
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_selective_loading_gpu(self, tmp_path):
        """Test selective column loading on GPU."""
        n = 1000
        data = {
            "text": np.ones((n, 768), dtype=np.float32) * 1.0,
            "image": np.ones((n, 512), dtype=np.float32) * 2.0,
            "audio": np.ones((n, 256), dtype=np.float32) * 3.0,
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path))
        
        # Load only text column on GPU
        loaded = load_cvc_columns(
            str(file_path),
            columns=["text"],
            device="cuda"
        )
        
        # Verify
        import torch
        assert set(loaded.keys()) == {"text"}
        assert loaded["text"].is_cuda
        assert torch.allclose(loaded["text"], torch.ones(n, 768, device="cuda") * 1.0, rtol=0.01)
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_selective_loading_two_columns_gpu(self, tmp_path):
        """Test loading 2 of 4 columns on GPU."""
        n = 500
        data = {
            "col1": np.random.randn(n, 128).astype(np.float32),
            "col2": np.random.randn(n, 256).astype(np.float32),
            "col3": np.random.randn(n, 512).astype(np.float32),
            "col4": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path))
        
        # Load col1 and col3 on GPU
        loaded = load_cvc_columns(
            str(file_path),
            columns=["col1", "col3"],
            device="cuda"
        )
        
        # Verify
        assert set(loaded.keys()) == {"col1", "col3"}
        import torch
        assert all(v.is_cuda for v in loaded.values())
        assert loaded["col1"].shape == (n, 128)
        assert loaded["col3"].shape == (n, 512)


class TestGPUBackendPerformance:
    """Performance-related tests for GPU backend."""
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_large_file_gpu(self, tmp_path):
        """Test GPU backend with larger file."""
        n = 10_000  # 10k vectors
        data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
        }
        
        file_path = tmp_path / "large.cvc"
        pack_cvc_columns(data, str(file_path), chunk_size=1_000)
        
        # Load on GPU
        loaded = load_cvc_columns(str(file_path), device="cuda")
        
        # Verify
        import torch
        assert loaded["text"].is_cuda
        assert loaded["text"].shape == (n, 768)
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU backend available")
    def test_gpu_vs_cpu_accuracy(self, tmp_path):
        """Test that GPU and CPU backends produce same results."""
        n = 1000
        data = {
            "embeddings": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(file_path))
        
        # Load on CPU
        cpu_result = load_cvc_columns(str(file_path), device="cpu")
        
        # Load on GPU
        gpu_result = load_cvc_columns(str(file_path), device="cuda")
        
        # Compare (GPU result needs to be moved to CPU)
        import torch
        gpu_cpu = gpu_result["embeddings"].cpu().numpy()
        
        np.testing.assert_allclose(
            cpu_result["embeddings"],
            gpu_cpu,
            rtol=0.01,
            atol=0.01
        )


if __name__ == "__main__":
    if not HAS_ANY_GPU:
        print("⚠️  No GPU backend available. Skipping GPU tests.")
        print(f"   CUDA available: {HAS_CUDA}")
        print(f"   Triton available: {HAS_TRITON}")
    else:
        print(f"✅ Running GPU tests")
        print(f"   CUDA available: {HAS_CUDA}")
        print(f"   Triton available: {HAS_TRITON}")
        pytest.main([__file__, "-v", "-s"])
