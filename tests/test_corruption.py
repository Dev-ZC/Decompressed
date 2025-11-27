"""Tests for corruption detection and recovery in CVC format."""

import numpy as np
import tempfile
import os
from pathlib import Path
import warnings

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from decompressed import (
    pack_cvc, 
    load_cvc, 
    validate_cvc_integrity, 
    CorruptedChunkError
)


class TestCorruption:
    """Test corruption detection and graceful degradation."""
    
    def test_intact_file_passes_integrity_check(self):
        """Intact files pass integrity validation."""
        vectors = np.random.randn(5000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "intact.cvc")
            pack_cvc(vectors, filepath, compression="fp16", chunk_size=1000)
            
            report = validate_cvc_integrity(filepath)
            
            assert report['valid'], f"Intact file should pass: {report}"
            assert report['total_chunks'] == 5
            assert len(report['corrupted_chunks']) == 0
            assert len(report['corrupted_ranges']) == 0
            
            print(f"✓ Intact file passes integrity check ({report['total_chunks']} chunks)")
    
    def test_corrupted_chunk_detected(self):
        """Corrupted chunks are detected by checksum validation."""
        vectors = np.random.randn(3000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "corrupted.cvc")
            pack_cvc(vectors, filepath, compression="fp16", chunk_size=1000)
            
            # Corrupt the file by flipping bits in the middle of chunk 1
            with open(filepath, "rb") as f:
                data = bytearray(f.read())
            
            # Find approximate location of chunk 1 payload
            # Header: 4 (magic) + 4 (header_len) + ~200 (json header)
            # Chunk 0: 4 (len) + 4 (checksum) + ~1.5MB payload
            corruption_offset = 220 + 8 + 1000 * 768 * 2 + 8 + 1000  # Middle of chunk 1
            
            # Flip some bits
            for i in range(10):
                data[corruption_offset + i] ^= 0xFF
            
            # Write corrupted data
            with open(filepath, "wb") as f:
                f.write(data)
            
            # Validate integrity
            report = validate_cvc_integrity(filepath)
            
            assert not report['valid'], "Corrupted file should fail validation"
            assert len(report['corrupted_chunks']) > 0, "Should detect corrupted chunks"
            
            print(f"✓ Corruption detected: chunk(s) {report['corrupted_chunks']}")
    
    def test_load_corrupted_file_raises_error(self):
        """Loading corrupted file raises CorruptedChunkError by default."""
        vectors = np.random.randn(2000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "corrupted.cvc")
            pack_cvc(vectors, filepath, compression="int8", chunk_size=1000)
            
            # Corrupt the file
            with open(filepath, "rb") as f:
                data = bytearray(f.read())
            
            # Corrupt chunk 0
            corruption_offset = 220 + 8 + 500
            for i in range(10):
                data[corruption_offset + i] ^= 0xFF
            
            with open(filepath, "wb") as f:
                f.write(data)
            
            # Try to load - should raise error
            try:
                loaded = load_cvc(filepath, device="cpu")
                assert False, "Should have raised CorruptedChunkError"
            except CorruptedChunkError as e:
                error_msg = str(e)
                assert "corrupted" in error_msg.lower()
                assert "CRC32" in error_msg
                print(f"✓ CorruptedChunkError raised correctly: {error_msg[:80]}...")
    
    def test_graceful_degradation_skip(self):
        """on_corruption='skip' fills corrupted chunks with fill_value."""
        vectors = np.random.randn(3000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "corrupted.cvc")
            pack_cvc(vectors, filepath, compression="fp16", chunk_size=1000)
            
            # Corrupt chunk 1
            with open(filepath, "rb") as f:
                data = bytearray(f.read())
            
            corruption_offset = 220 + 8 + 1000 * 768 * 2 + 8 + 500
            for i in range(100):
                data[corruption_offset + i] ^= 0xFF
            
            with open(filepath, "wb") as f:
                f.write(data)
            
            # Load with graceful degradation
            loaded = load_cvc(filepath, device="cpu", on_corruption="skip", fill_value=0.0)
            
            # Should have loaded successfully
            assert loaded.shape == (3000, 768)
            
            # Chunk 1 (vectors 1000-2000) should be all zeros
            chunk1 = loaded[1000:2000, :]
            assert np.all(chunk1 == 0.0), "Corrupted chunk should be filled with 0.0"
            
            # Other chunks should have non-zero data
            chunk0_nonzero = np.count_nonzero(loaded[0:1000, :])
            assert chunk0_nonzero > 0, "Intact chunks should have data"
            
            print(f"✓ Graceful degradation (skip): corrupted chunk filled with 0.0")
    
    def test_graceful_degradation_warn(self):
        """on_corruption='warn' fills corrupted chunks and logs warnings."""
        vectors = np.random.randn(2000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "corrupted.cvc")
            pack_cvc(vectors, filepath, compression="int8", chunk_size=1000)
            
            # Corrupt chunk 0
            with open(filepath, "rb") as f:
                data = bytearray(f.read())
            
            corruption_offset = 220 + 8 + 500
            for i in range(50):
                data[corruption_offset + i] ^= 0xFF
            
            with open(filepath, "wb") as f:
                f.write(data)
            
            # Load with warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded = load_cvc(filepath, device="cpu", on_corruption="warn", fill_value=-999.0)
                
                # Should have warned
                assert len(w) > 0, "Should have issued warning"
                warning_msg = str(w[0].message)
                assert "corrupted" in warning_msg.lower()
                assert "-999.0" in warning_msg
            
            # Chunk 0 should be filled with -999.0
            chunk0 = loaded[0:1000, :]
            assert np.all(chunk0 == -999.0), "Corrupted chunk should be filled with -999.0"
            
            print(f"✓ Graceful degradation (warn): warning issued and chunk filled")
    
    def test_multiple_corrupted_chunks(self):
        """Multiple corrupted chunks are all detected."""
        vectors = np.random.randn(5000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "multi_corrupt.cvc")
            pack_cvc(vectors, filepath, compression="fp16", chunk_size=1000)
            
            # Corrupt chunks 0, 2, and 4
            with open(filepath, "rb") as f:
                data = bytearray(f.read())
            
            # Corrupt chunk 0
            offset0 = 220 + 8 + 500
            for i in range(10):
                data[offset0 + i] ^= 0xFF
            
            # Corrupt chunk 2  
            offset2 = 220 + (8 + 1000 * 768 * 2) * 2 + 8 + 500
            for i in range(10):
                data[offset2 + i] ^= 0xFF
            
            # Corrupt chunk 4
            offset4 = 220 + (8 + 1000 * 768 * 2) * 4 + 8 + 500
            for i in range(10):
                data[offset4 + i] ^= 0xFF
            
            with open(filepath, "wb") as f:
                f.write(data)
            
            # Validate integrity
            report = validate_cvc_integrity(filepath)
            
            assert not report['valid']
            assert len(report['corrupted_chunks']) == 3
            assert 0 in report['corrupted_chunks']
            assert 2 in report['corrupted_chunks']
            assert 4 in report['corrupted_chunks']
            
            # Check corrupted ranges
            assert len(report['corrupted_ranges']) == 3
            assert (0, 1000) in report['corrupted_ranges']
            assert (2000, 3000) in report['corrupted_ranges']
            assert (4000, 5000) in report['corrupted_ranges']
            
            print(f"✓ Multiple corruptions detected: {report['corrupted_chunks']}")
    
    def test_integrity_check_verbose_mode(self):
        """Verbose mode provides detailed checksum information."""
        vectors = np.random.randn(2000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.cvc")
            pack_cvc(vectors, filepath, compression="fp16", chunk_size=1000)
            
            # Run with verbose=True (capture output)
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                report = validate_cvc_integrity(filepath, verbose=True)
            
            output = f.getvalue()
            
            assert report['valid']
            assert report['checksum_details'] is not None
            assert len(report['checksum_details']) == 2
            
            # Check details structure
            for detail in report['checksum_details']:
                assert 'chunk_index' in detail
                assert 'valid' in detail
                assert 'expected_crc32' in detail
                assert 'actual_crc32' in detail
                assert 'vector_range' in detail
            
            # Check output
            assert "Validating 2 chunks" in output
            assert "✓ Chunk 0: OK" in output
            assert "✓ Chunk 1: OK" in output
            
            print(f"✓ Verbose mode provides detailed information")


def run_all_tests():
    """Run all corruption tests."""
    test_suite = TestCorruption()
    
    tests = [
        ("Intact File Integrity Check", test_suite.test_intact_file_passes_integrity_check),
        ("Corrupted Chunk Detection", test_suite.test_corrupted_chunk_detected),
        ("Load Corrupted Raises Error", test_suite.test_load_corrupted_file_raises_error),
        ("Graceful Degradation (Skip)", test_suite.test_graceful_degradation_skip),
        ("Graceful Degradation (Warn)", test_suite.test_graceful_degradation_warn),
        ("Multiple Corrupted Chunks", test_suite.test_multiple_corrupted_chunks),
        ("Verbose Integrity Check", test_suite.test_integrity_check_verbose_mode),
    ]
    
    print("\n" + "="*70)
    print("PHASE 2 CORRUPTION DETECTION TESTS")
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
