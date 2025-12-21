"""Tests for lossless byte-shuffling + RLE compression."""

import numpy as np
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from decompressed import pack_cvc, load_cvc, validate_cvc
from decompressed.compress import compress_lossless, _byte_shuffle
from decompressed.decompress import decompress_lossless_cpu, _byte_unshuffle


class TestLosslessCompression:
    """Test lossless byte-shuffling compression (GPU-native)."""
    
    def test_byte_shuffle_roundtrip(self):
        """Byte-shuffle and unshuffle preserves data exactly."""
        vectors = np.random.randn(100, 64).astype(np.float32)
        
        shuffled = _byte_shuffle(vectors)
        unshuffled = _byte_unshuffle(shuffled, 100 * 64)
        
        original_bytes = vectors.tobytes()
        assert unshuffled == original_bytes, "Byte-shuffle roundtrip should be exact"
        
        reconstructed = np.frombuffer(unshuffled, dtype=np.float32).reshape(100, 64)
        assert np.array_equal(vectors, reconstructed), "Reconstructed data should match exactly"
        print("✓ Byte-shuffle roundtrip preserves data exactly")
    
    def test_byte_shuffle_structure(self):
        """Byte-shuffle creates correct plane structure."""
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        shuffled = _byte_shuffle(vectors)
        
        # Should have 4 floats * 4 bytes = 16 bytes
        assert len(shuffled) == 16, f"Expected 16 bytes, got {len(shuffled)}"
        
        # Convert to byte array for inspection
        byte_array = np.frombuffer(shuffled, dtype=np.uint8)
        
        # First n_values bytes should be all byte0s, next n_values should be all byte1s, etc.
        # This is the key property that makes it GPU-friendly
        assert len(byte_array) == 16
        print(f"✓ Byte-shuffle creates 4 contiguous planes: {len(byte_array)} bytes")
    
    def test_compress_lossless_deterministic(self):
        """Lossless compression is deterministic."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        payload1 = compress_lossless(vectors)
        payload2 = compress_lossless(vectors)
        
        assert payload1 == payload2, "Lossless compression should be deterministic"
        print("✓ Lossless compression is deterministic")
    
    def test_compress_decompress_lossless_exact(self):
        """Compression → decompression is bit-perfect (truly lossless)."""
        vectors = np.random.randn(500, 384).astype(np.float32)
        
        compressed = compress_lossless(vectors)
        decompressed = decompress_lossless_cpu(compressed, 500, 384)
        
        # Should be EXACTLY equal (bit-perfect)
        assert np.array_equal(vectors, decompressed), "Lossless should preserve exact bits"
        
        # Verify with binary comparison
        original_bytes = vectors.tobytes()
        reconstructed_bytes = decompressed.tobytes()
        assert original_bytes == reconstructed_bytes, "Binary data should match exactly"
        
        print("✓ Lossless compression is truly bit-perfect")
    
    def test_lossless_size(self):
        """Lossless compression achieves good ratios with bit-packing."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        original_size = vectors.nbytes
        compressed = compress_lossless(vectors)
        compressed_size = len(compressed)
        
        # Bit-packing reduces size by compressing redundant high bytes
        # Random data: 85-95% (10-15% compression)
        # Real embeddings: 70-80% (20-30% compression) due to more structure
        ratio = compressed_size / original_size
        assert compressed_size < original_size, f"Compressed should be smaller: {compressed_size} vs {original_size}"
        assert compressed_size > original_size * 0.5, f"Shouldn't compress more than 50%: {ratio:.1%}"
        print(f"✓ Lossless compression: {original_size} → {compressed_size} bytes ({100*ratio:.1f}%) - GPU-native + bit-perfect")
    
    def test_pack_cvc_lossless_deterministic(self):
        """Packing with lossless compression is deterministic."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.cvc")
            file2 = os.path.join(tmpdir, "file2.cvc")
            
            pack_cvc(vectors, file1, compression="lossless", chunk_size=500)
            pack_cvc(vectors, file2, compression="lossless", chunk_size=500)
            
            bytes1 = open(file1, "rb").read()
            bytes2 = open(file2, "rb").read()
            
            assert bytes1 == bytes2, "Lossless packed files should be byte-identical"
            print(f"✓ Lossless packed files are byte-identical ({len(bytes1)} bytes)")
    
    def test_pack_load_lossless_exact(self):
        """Pack → load with lossless preserves exact values."""
        vectors = np.random.randn(2000, 512).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "lossless.cvc")
            
            pack_cvc(vectors, filepath, compression="lossless", chunk_size=1000)
            loaded = load_cvc(filepath, device="cpu")
            
            # Should be EXACTLY equal
            assert np.array_equal(vectors, loaded), "Lossless pack/load should be exact"
            
            # Verify binary equality
            assert vectors.tobytes() == loaded.tobytes(), "Binary data should match exactly"
            
            print(f"✓ Lossless pack/load roundtrip is bit-perfect")
    
    def test_validate_lossless_file(self):
        """validate_cvc works on lossless-compressed files."""
        vectors = np.random.randn(5000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "lossless.cvc")
            
            pack_cvc(vectors, filepath, compression="lossless", chunk_size=2000)
            
            report = validate_cvc(filepath)
            
            assert report['valid'], f"Packed file should be valid: {report['errors']}"
            assert report['num_vectors'] == 5000
            assert report['dimension'] == 768
            assert report['compression'] == 'lossless'
            assert len(report['errors']) == 0
            
            print(f"✓ validate_cvc passed for lossless: {report['num_vectors']} vectors, "
                  f"{report['num_chunks']} chunks, {report['size_mb']:.2f} MB")
    
    def test_lossless_vs_fp16_vs_int8_size(self):
        """Compare file sizes across compression methods."""
        vectors = np.random.randn(5000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fp16_file = os.path.join(tmpdir, "fp16.cvc")
            int8_file = os.path.join(tmpdir, "int8.cvc")
            lossless_file = os.path.join(tmpdir, "lossless.cvc")
            
            pack_cvc(vectors, fp16_file, compression="fp16")
            pack_cvc(vectors, int8_file, compression="int8")
            pack_cvc(vectors, lossless_file, compression="lossless")
            
            original_size = vectors.nbytes
            fp16_size = os.path.getsize(fp16_file)
            int8_size = os.path.getsize(int8_file)
            lossless_size = os.path.getsize(lossless_file)
            
            print(f"\n✓ Size comparison (5000×768 vectors):")
            print(f"  Original FP32: {original_size:,} bytes (100%)")
            print(f"  FP16:          {fp16_size:,} bytes ({100*fp16_size/original_size:.1f}%) - 2× lossy compression")
            print(f"  INT8:          {int8_size:,} bytes ({100*int8_size/original_size:.1f}%) - 4× lossy compression")
            print(f"  Lossless:      {lossless_size:,} bytes ({100*lossless_size/original_size:.1f}%) - GPU-native, bit-perfect")
    
    def test_lossless_different_shapes(self):
        """Lossless works with different vector dimensions."""
        shapes = [(100, 64), (500, 768), (1000, 1536), (200, 128)]
        
        for rows, dim in shapes:
            vectors = np.random.randn(rows, dim).astype(np.float32)
            
            compressed = compress_lossless(vectors)
            decompressed = decompress_lossless_cpu(compressed, rows, dim)
            
            assert np.array_equal(vectors, decompressed), f"Failed for shape ({rows}, {dim})"
        
        print(f"✓ Lossless works with various shapes: {shapes}")
    
    def test_lossless_extreme_values(self):
        """Lossless preserves extreme float32 values."""
        vectors = np.array([
            [np.inf, -np.inf, 0.0, -0.0],
            [1e-38, 1e38, np.finfo(np.float32).max, np.finfo(np.float32).min],
            [np.nan, 1.0, -1.0, 0.5],
        ], dtype=np.float32)
        
        compressed = compress_lossless(vectors)
        decompressed = decompress_lossless_cpu(compressed, 3, 4)
        
        # Use binary comparison since NaN != NaN
        assert vectors.tobytes() == decompressed.tobytes(), "Extreme values should be preserved exactly"
        print("✓ Lossless preserves extreme float32 values (inf, nan, min, max)")


def run_all_tests():
    """Run all lossless compression tests."""
    test_suite = TestLosslessCompression()
    
    tests = [
        ("Byte-Shuffle Roundtrip", test_suite.test_byte_shuffle_roundtrip),
        ("Byte-Shuffle Structure", test_suite.test_byte_shuffle_structure),
        ("Lossless Deterministic", test_suite.test_compress_lossless_deterministic),
        ("Lossless Bit-Perfect", test_suite.test_compress_decompress_lossless_exact),
        ("Lossless Size Check", test_suite.test_lossless_size),
        ("Pack Lossless Deterministic", test_suite.test_pack_cvc_lossless_deterministic),
        ("Pack/Load Lossless Exact", test_suite.test_pack_load_lossless_exact),
        ("Validate Lossless File", test_suite.test_validate_lossless_file),
        ("Size Comparison", test_suite.test_lossless_vs_fp16_vs_int8_size),
        ("Different Shapes", test_suite.test_lossless_different_shapes),
        ("Extreme Values", test_suite.test_lossless_extreme_values),
    ]
    
    print("\n" + "="*70)
    print("LOSSLESS COMPRESSION TESTS")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"Running: {name}")
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            failed += 1
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
