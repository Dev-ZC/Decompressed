"""Tests for memory-mapped CVC loading."""

import numpy as np
import pytest
from pathlib import Path

from decompressed import pack_cvc, MMapCVCLoader


class TestMMapBasic:
    """Basic memory-mapped loading tests."""
    
    def test_mmap_open_close(self, tmp_path):
        """Test opening and closing mmap file."""
        # Create test file
        vectors = np.random.randn(1000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), mmap_optimized=True)
        
        # Open with mmap
        loader = MMapCVCLoader(str(file_path))
        loader.open()
        
        assert loader.mmap is not None
        assert loader.header is not None
        assert loader.is_mmap_optimized == True
        
        loader.close()
        assert loader.mmap is None
    
    def test_mmap_context_manager(self, tmp_path):
        """Test mmap with context manager."""
        vectors = np.random.randn(500, 128).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), mmap_optimized=True)
        
        # Use context manager
        with MMapCVCLoader(str(file_path)) as loader:
            assert loader.mmap is not None
            info = loader.get_info()
            assert info["num_vectors"] == 500
            assert info["dimension"] == 128
            assert info["mmap_optimized"] == True
        
        # Verify closed
        assert loader.mmap is None
    
    def test_mmap_optimized_flag(self, tmp_path):
        """Test that mmap_optimized flag is detected."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        # Without optimization
        file1 = tmp_path / "regular.cvc"
        pack_cvc(vectors, str(file1), mmap_optimized=False)
        
        with MMapCVCLoader(str(file1)) as loader:
            assert loader.is_mmap_optimized == False
        
        # With optimization
        file2 = tmp_path / "optimized.cvc"
        pack_cvc(vectors, str(file2), mmap_optimized=True)
        
        with MMapCVCLoader(str(file2)) as loader:
            assert loader.is_mmap_optimized == True


class TestMMapChunkAccess:
    """Test zero-copy chunk access."""
    
    def test_get_chunk_view(self, tmp_path):
        """Test getting zero-copy chunk view."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), chunk_size=100, mmap_optimized=True)
        
        with MMapCVCLoader(str(file_path)) as loader:
            # Get view of first chunk
            view = loader.get_chunk_view(0)
            
            # Verify it's a memoryview (zero-copy)
            assert isinstance(view, memoryview)
            assert len(view) > 0
    
    def test_load_single_chunk(self, tmp_path):
        """Test loading single chunk via mmap."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), chunk_size=100, mmap_optimized=True)
        
        with MMapCVCLoader(str(file_path)) as loader:
            # Load first chunk
            chunk = loader.load_chunk(0, device="cpu")
            
            assert chunk.shape == (100, 768)
            # Verify approximate accuracy (FP16 compression)
            np.testing.assert_allclose(chunk, vectors[:100], rtol=0.01, atol=0.01)


class TestMMapFullLoad:
    """Test loading entire file with mmap."""
    
    def test_mmap_load_full_file(self, tmp_path):
        """Test loading full file via mmap."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), mmap_optimized=True)
        
        with MMapCVCLoader(str(file_path)) as loader:
            loaded = loader.load(device="cpu")
            
            assert loaded.shape == (1000, 768)
            np.testing.assert_allclose(loaded, vectors, rtol=0.01, atol=0.01)
    
    def test_mmap_load_chunked_file(self, tmp_path):
        """Test loading chunked file via mmap."""
        vectors = np.random.randn(5000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), chunk_size=500, mmap_optimized=True)
        
        with MMapCVCLoader(str(file_path)) as loader:
            loaded = loader.load(device="cpu")
            
            assert loaded.shape == (5000, 768)
            np.testing.assert_allclose(loaded, vectors, rtol=0.01, atol=0.01)


class TestMMapCompression:
    """Test mmap with different compressions."""
    
    def test_mmap_fp16_compression(self, tmp_path):
        """Test mmap with FP16 compression."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), compression="fp16", mmap_optimized=True)
        
        with MMapCVCLoader(str(file_path)) as loader:
            loaded = loader.load()
            np.testing.assert_allclose(loaded, vectors, rtol=0.01, atol=0.01)
    
    def test_mmap_int8_compression(self, tmp_path):
        """Test mmap with INT8 compression."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), compression="int8", mmap_optimized=True)
        
        with MMapCVCLoader(str(file_path)) as loader:
            loaded = loader.load()
            # INT8 has lower accuracy
            np.testing.assert_allclose(loaded, vectors, rtol=0.1, atol=0.1)


class TestMMapNonOptimized:
    """Test mmap with non-optimized files (fallback)."""
    
    def test_mmap_non_optimized_file(self, tmp_path):
        """Test that mmap works with non-optimized files (slower)."""
        vectors = np.random.randn(500, 128).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), mmap_optimized=False)  # No optimization
        
        with MMapCVCLoader(str(file_path)) as loader:
            # Should still work, just slower
            assert loader.is_mmap_optimized == False
            
            loaded = loader.load()
            assert loaded.shape == (500, 128)
            np.testing.assert_allclose(loaded, vectors, rtol=0.01, atol=0.01)


class TestMMapInfo:
    """Test mmap info retrieval."""
    
    def test_get_info(self, tmp_path):
        """Test getting file info from mmap."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        file_path = tmp_path / "test.cvc"
        pack_cvc(vectors, str(file_path), chunk_size=100, mmap_optimized=True)
        
        with MMapCVCLoader(str(file_path)) as loader:
            info = loader.get_info()
            
            assert info["num_vectors"] == 1000
            assert info["dimension"] == 768
            assert info["num_chunks"] == 10
            assert info["mmap_optimized"] == True
            assert info["file_size"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
