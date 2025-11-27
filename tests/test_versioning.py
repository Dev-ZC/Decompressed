"""Tests for format versioning and migration."""

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
    upgrade_cvc,
    detect_version,
    validate_cvc_version,
    validate_cvc,
)


def create_legacy_v0_file(path, vectors):
    """Create a v0.x CVC file (legacy format without version bytes or checksums)."""
    import json
    import zlib
    from decompressed.compress import compress_fp16
    
    HEADER_MAGIC = b"CVCF"
    n_vectors, dim = vectors.shape
    chunk_size = 1000
    
    # Build chunks (v0.x style - no checksums)
    chunks_meta = []
    chunk_payloads = []
    
    for start_idx in range(0, n_vectors, chunk_size):
        end_idx = min(start_idx + chunk_size, n_vectors)
        chunk_vectors = vectors[start_idx:end_idx]
        rows = end_idx - start_idx
        
        payload = compress_fp16(chunk_vectors)
        chunk_meta = {"rows": rows, "compression": "fp16"}
        
        chunks_meta.append(chunk_meta)
        chunk_payloads.append(payload)
    
    # Build header (v0.x style)
    header = {
        "num_vectors": n_vectors,
        "dimension": dim,
        "compression": "fp16",
        "chunks": chunks_meta
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(',', ':')).encode('utf-8')
    header_len = len(header_bytes)
    
    # Write v0.x file (no version bytes, no checksums)
    with open(path, "wb") as f:
        f.write(HEADER_MAGIC)
        f.write(header_len.to_bytes(4, byteorder='little'))
        f.write(header_bytes)
        
        for payload in chunk_payloads:
            f.write(len(payload).to_bytes(4, byteorder='little'))
            f.write(payload)


class TestVersioning:
    """Test format versioning and migration."""
    
    def test_new_files_are_v1(self):
        """Newly packed files should be v1.0."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "new.cvc")
            pack_cvc(vectors, filepath, compression="fp16")
            
            version = detect_version(filepath)
            assert version == (1, 0), f"New files should be v1.0, got v{version[0]}.{version[1]}"
            
            info = validate_cvc_version(filepath)
            assert info['version'] == "v1.0"
            assert not info['needs_upgrade']
            
            print(f"✓ New files are v1.0")
    
    def test_detect_legacy_v0_file(self):
        """Detect legacy v0.x files correctly."""
        vectors = np.random.randn(2000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "legacy.cvc")
            create_legacy_v0_file(filepath, vectors)
            
            version = detect_version(filepath)
            assert version == (0, 1), f"Legacy files should be v0.1, got v{version[0]}.{version[1]}"
            
            info = validate_cvc_version(filepath)
            assert info['version'] == "v0.1"
            assert info['needs_upgrade']
            
            print(f"✓ Legacy v0.x files detected correctly")
    
    def test_load_legacy_v0_file_with_warning(self):
        """Loading v0.x files should work but issue deprecation warning."""
        vectors = np.random.randn(1500, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "legacy.cvc")
            create_legacy_v0_file(filepath, vectors)
            
            # Load with warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded = load_cvc(filepath, device="cpu")
                
                # Should have warned about legacy format
                assert len(w) > 0, "Should have issued deprecation warning"
                warning_msg = str(w[0].message)
                assert "legacy" in warning_msg.lower() or "v0" in warning_msg.lower()
            
            # Data should still load correctly
            assert loaded.shape == vectors.shape
            
            print(f"✓ Legacy v0.x files load with deprecation warning")
    
    def test_upgrade_v0_to_v1(self):
        """upgrade_cvc migrates v0.x to v1.0."""
        vectors = np.random.randn(3000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = os.path.join(tmpdir, "old.cvc")
            new_file = os.path.join(tmpdir, "new.cvc")
            
            # Create v0.x file
            create_legacy_v0_file(old_file, vectors)
            assert detect_version(old_file) == (0, 1)
            
            # Upgrade
            report = upgrade_cvc(old_file, new_file, verbose=False)
            
            assert report['input_version'] == "v0.1"
            assert report['output_version'] == "v1.0"
            assert report['num_vectors'] == 3000
            assert report['dimension'] == 768
            
            # Verify new file is v1.0
            assert detect_version(new_file) == (1, 0)
            
            # Data should be identical (within FP16 precision)
            old_data = load_cvc(old_file, device="cpu")
            new_data = load_cvc(new_file, device="cpu")
            
            max_diff = np.abs(old_data - new_data).max()
            assert max_diff < 1e-10, f"Data should be identical, max diff: {max_diff}"
            
            print(f"✓ upgrade_cvc migrates v0.x → v1.0")
    
    def test_upgrade_v1_to_v1_is_noop(self):
        """Upgrading v1.0 file rewrites it deterministically."""
        vectors = np.random.randn(2000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.cvc")
            file2 = os.path.join(tmpdir, "file2.cvc")
            
            # Create v1.0 file
            pack_cvc(vectors, file1, compression="fp16")
            assert detect_version(file1) == (1, 0)
            
            # "Upgrade" v1.0 → v1.0
            report = upgrade_cvc(file1, file2)
            
            assert report['input_version'] == "v1.0"
            assert report['output_version'] == "v1.0"
            
            # Files should be identical
            bytes1 = open(file1, "rb").read()
            bytes2 = open(file2, "rb").read()
            assert bytes1 == bytes2, "v1.0 → v1.0 should produce identical file"
            
            print(f"✓ v1.0 → v1.0 upgrade produces identical file")
    
    def test_validate_cvc_detects_version(self):
        """validate_cvc should report correct version."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test v1.0 file
            v1_file = os.path.join(tmpdir, "v1.cvc")
            pack_cvc(vectors, v1_file, compression="fp16")
            
            report = validate_cvc(v1_file)
            assert report['valid']
            assert report['version'] == "v1.0"
            
            # Test v0.1 file
            v0_file = os.path.join(tmpdir, "v0.cvc")
            create_legacy_v0_file(v0_file, vectors)
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")  # Suppress warnings for test
                report = validate_cvc(v0_file)
            
            assert report['valid']
            assert report['version'] == "v0.1"
            assert len(report['warnings']) > 0  # Should warn about legacy format
            
            print(f"✓ validate_cvc detects versions correctly")
    
    def test_roundtrip_v1_preserves_data(self):
        """Pack→Load roundtrip with v1.0 preserves data."""
        vectors = np.random.randn(2000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "roundtrip.cvc")
            
            pack_cvc(vectors, filepath, compression="fp16", chunk_size=500)
            assert detect_version(filepath) == (1, 0)
            
            loaded = load_cvc(filepath, device="cpu")
            
            # Should be very close (FP16 precision)
            max_error = np.abs(vectors - loaded).max()
            assert max_error < 0.01
            
            print(f"✓ v1.0 roundtrip preserves data (max error: {max_error:.6f})")
    
    def test_upgrade_preserves_checksums(self):
        """Upgrading v0.x adds checksums that validate correctly."""
        from decompressed import validate_cvc_integrity
        
        vectors = np.random.randn(2000, 768).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = os.path.join(tmpdir, "old.cvc")
            new_file = os.path.join(tmpdir, "new.cvc")
            
            # Create v0.x file (no checksums)
            create_legacy_v0_file(old_file, vectors)
            
            # v0.x files can't be integrity-checked
            old_integrity = validate_cvc_integrity(old_file)
            assert not old_integrity['valid']
            assert "no checksums" in old_integrity['error'].lower()
            
            # Upgrade
            upgrade_cvc(old_file, new_file)
            
            # v1.0 file should pass integrity check
            new_integrity = validate_cvc_integrity(new_file)
            assert new_integrity['valid']
            assert len(new_integrity['corrupted_chunks']) == 0
            
            print(f"✓ Upgraded files have valid checksums")


def run_all_tests():
    """Run all versioning tests."""
    test_suite = TestVersioning()
    
    tests = [
        ("New Files Are v1.0", test_suite.test_new_files_are_v1),
        ("Detect Legacy v0.x Files", test_suite.test_detect_legacy_v0_file),
        ("Load Legacy Files With Warning", test_suite.test_load_legacy_v0_file_with_warning),
        ("Upgrade v0 → v1", test_suite.test_upgrade_v0_to_v1),
        ("Upgrade v1 → v1 (No-op)", test_suite.test_upgrade_v1_to_v1_is_noop),
        ("Validate CVC Detects Version", test_suite.test_validate_cvc_detects_version),
        ("Roundtrip v1.0 Preserves Data", test_suite.test_roundtrip_v1_preserves_data),
        ("Upgrade Preserves Checksums", test_suite.test_upgrade_preserves_checksums),
    ]
    
    print("\n" + "="*70)
    print("PHASE 3 VERSIONING & MIGRATION TESTS")
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
