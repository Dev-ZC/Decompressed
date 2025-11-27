"""Tests for deterministic reproducibility of CVC format."""

import numpy as np
import tempfile
import os
from pathlib import Path

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from decompressed import pack_cvc, pack_cvc_sections, load_cvc, validate_cvc
from decompressed.compress import compress_fp16, compress_int8


class TestReproducibility:
    """Test that CVC format is deterministic and reproducible."""
    
    def test_compress_fp16_deterministic(self):
        """FP16 compression produces identical bytes for same input."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        payload1 = compress_fp16(vectors)
        payload2 = compress_fp16(vectors)
        
        assert payload1 == payload2, "FP16 compression should be deterministic"
        print("✓ FP16 compression is deterministic")
    
    def test_compress_int8_deterministic(self):
        """INT8 compression produces identical bytes for same input."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        payload1, min1, scale1 = compress_int8(vectors)
        payload2, min2, scale2 = compress_int8(vectors)
        
        assert payload1 == payload2, "INT8 compression payload should be deterministic"
        assert min1 == min2, f"INT8 min should be deterministic: {min1} vs {min2}"
        assert scale1 == scale2, f"INT8 scale should be deterministic: {scale1} vs {scale2}"
        print(f"✓ INT8 compression is deterministic (min={min1:.6f}, scale={scale1:.6f})")
    
    def test_pack_cvc_fp16_deterministic(self):
        """Packing same data twice produces identical file bytes (FP16)."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.cvc")
            file2 = os.path.join(tmpdir, "file2.cvc")
            
            pack_cvc(vectors, file1, compression="fp16", chunk_size=500)
            pack_cvc(vectors, file2, compression="fp16", chunk_size=500)
            
            bytes1 = open(file1, "rb").read()
            bytes2 = open(file2, "rb").read()
            
            assert bytes1 == bytes2, "FP16 packed files should be byte-identical"
            print(f"✓ FP16 packed files are byte-identical ({len(bytes1)} bytes)")
    
    def test_pack_cvc_int8_deterministic(self):
        """Packing same data twice produces identical file bytes (INT8)."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.cvc")
            file2 = os.path.join(tmpdir, "file2.cvc")
            
            pack_cvc(vectors, file1, compression="int8", chunk_size=500)
            pack_cvc(vectors, file2, compression="int8", chunk_size=500)
            
            bytes1 = open(file1, "rb").read()
            bytes2 = open(file2, "rb").read()
            
            assert bytes1 == bytes2, "INT8 packed files should be byte-identical"
            print(f"✓ INT8 packed files are byte-identical ({len(bytes1)} bytes)")
    
    def test_pack_cvc_sections_deterministic(self):
        """pack_cvc_sections produces deterministic output."""
        wiki = np.random.randn(1000, 768).astype(np.float32)
        arxiv = np.random.randn(1500, 768).astype(np.float32)
        
        sections = [
            (wiki, {"source": "wikipedia", "date": "2024-01"}),
            (arxiv, {"source": "arxiv", "date": "2024-02"}),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "sections1.cvc")
            file2 = os.path.join(tmpdir, "sections2.cvc")
            
            pack_cvc_sections(sections, file1, compression="fp16", chunk_size=800)
            pack_cvc_sections(sections, file2, compression="fp16", chunk_size=800)
            
            bytes1 = open(file1, "rb").read()
            bytes2 = open(file2, "rb").read()
            
            assert bytes1 == bytes2, "pack_cvc_sections should produce byte-identical files"
            print(f"✓ pack_cvc_sections is byte-identical ({len(bytes1)} bytes)")
    
    def test_different_data_produces_different_files(self):
        """Different data should produce different files (sanity check)."""
        vectors1 = np.random.randn(1000, 768).astype(np.float32)
        vectors2 = np.random.randn(1000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.cvc")
            file2 = os.path.join(tmpdir, "file2.cvc")
            
            pack_cvc(vectors1, file1, compression="fp16")
            pack_cvc(vectors2, file2, compression="fp16")
            
            bytes1 = open(file1, "rb").read()
            bytes2 = open(file2, "rb").read()
            
            assert bytes1 != bytes2, "Different data should produce different files"
            print("✓ Different data produces different files (as expected)")
    
    def test_validate_cvc_on_packed_file(self):
        """validate_cvc works on freshly packed files."""
        vectors = np.random.randn(10000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.cvc")
            
            pack_cvc(vectors, filepath, compression="int8", chunk_size=2500)
            
            report = validate_cvc(filepath)
            
            assert report['valid'], f"Packed file should be valid: {report['errors']}"
            assert report['num_vectors'] == 10000
            assert report['dimension'] == 768
            assert report['compression'] == 'int8'
            assert report['num_chunks'] == 4
            assert len(report['errors']) == 0
            # Version is v1.0 for new format
            assert report['version'] in ['0.1.0', 'v1.0']
            
            print(f"✓ validate_cvc passed: {report['num_vectors']} vectors, "
                  f"{report['num_chunks']} chunks, {report['size_mb']} MB")
    
    def test_roundtrip_preserves_data(self):
        """Data survives pack -> load roundtrip (with expected precision loss)."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test FP16
            fp16_file = os.path.join(tmpdir, "fp16.cvc")
            pack_cvc(vectors, fp16_file, compression="fp16")
            loaded_fp16 = load_cvc(fp16_file, device="cpu")
            
            # FP16 should have small error
            fp16_error = np.abs(vectors - loaded_fp16).max()
            assert fp16_error < 0.01, f"FP16 error too large: {fp16_error}"
            print(f"✓ FP16 roundtrip max error: {fp16_error:.6f}")
            
            # Test INT8
            int8_file = os.path.join(tmpdir, "int8.cvc")
            pack_cvc(vectors, int8_file, compression="int8")
            loaded_int8 = load_cvc(int8_file, device="cpu")
            
            # INT8 will have larger error but should still be reasonable
            int8_error = np.abs(vectors - loaded_int8).max()
            assert int8_error < 1.0, f"INT8 error too large: {int8_error}"
            print(f"✓ INT8 roundtrip max error: {int8_error:.6f}")


def run_all_tests():
    """Run all reproducibility tests."""
    test_suite = TestReproducibility()
    
    tests = [
        ("Compress FP16 Deterministic", test_suite.test_compress_fp16_deterministic),
        ("Compress INT8 Deterministic", test_suite.test_compress_int8_deterministic),
        ("Pack CVC FP16 Deterministic", test_suite.test_pack_cvc_fp16_deterministic),
        ("Pack CVC INT8 Deterministic", test_suite.test_pack_cvc_int8_deterministic),
        ("Pack CVC Sections Deterministic", test_suite.test_pack_cvc_sections_deterministic),
        ("Different Data Different Files", test_suite.test_different_data_produces_different_files),
        ("Validate CVC", test_suite.test_validate_cvc_on_packed_file),
        ("Roundtrip Data", test_suite.test_roundtrip_preserves_data),
    ]
    
    print("\n" + "="*70)
    print("PHASE 1 REPRODUCIBILITY TESTS")
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
