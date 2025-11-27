"""Tests for multi-column CVC functionality."""

import numpy as np
import tempfile
import pytest
from pathlib import Path

from decompressed import (
    pack_cvc_columns,
    pack_cvc_sections_columnar,
    load_cvc_columns
)


class TestColumnarBasic:
    """Basic multi-column functionality tests."""
    
    def test_pack_and_load_multiple_columns(self, tmp_path):
        """Test packing and loading multiple columns."""
        # Create test data
        n = 1000
        data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
            "doc_id": np.arange(n, dtype=np.int32)
        }
        
        output_file = tmp_path / "multi_column.cvc"
        
        # Pack
        pack_cvc_columns(
            data,
            str(output_file),
            compressions={
                "text": "fp16",
                "image": "int8",
                "doc_id": "none"
            },
            chunk_size=100
        )
        
        assert output_file.exists()
        
        # Load all columns
        loaded = load_cvc_columns(str(output_file))
        
        assert set(loaded.keys()) == {"text", "image", "doc_id"}
        assert loaded["text"].shape == (n, 768)
        assert loaded["image"].shape == (n, 512)
        assert loaded["doc_id"].shape == (n,)
        
        # Check approximate accuracy (fp16 and int8 are lossy)
        np.testing.assert_allclose(loaded["text"], data["text"], rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(loaded["image"], data["image"], rtol=0.1, atol=0.1)
        np.testing.assert_array_equal(loaded["doc_id"].astype(np.int32), data["doc_id"])
    
    def test_selective_column_loading(self, tmp_path):
        """Test loading only specific columns."""
        n = 500
        data = {
            "col1": np.random.randn(n, 128).astype(np.float32),
            "col2": np.random.randn(n, 256).astype(np.float32),
            "col3": np.random.randn(n, 64).astype(np.float32),
        }
        
        output_file = tmp_path / "selective.cvc"
        pack_cvc_columns(data, str(output_file))
        
        # Load only col1
        loaded = load_cvc_columns(str(output_file), columns=["col1"])
        
        assert set(loaded.keys()) == {"col1"}
        assert loaded["col1"].shape == (n, 128)
        np.testing.assert_allclose(loaded["col1"], data["col1"], rtol=0.01, atol=0.01)
        
        # Load col1 and col3
        loaded = load_cvc_columns(str(output_file), columns=["col1", "col3"])
        
        assert set(loaded.keys()) == {"col1", "col3"}
        np.testing.assert_allclose(loaded["col1"], data["col1"], rtol=0.01, atol=0.01)
        np.testing.assert_allclose(loaded["col3"], data["col3"], rtol=0.01, atol=0.01)
    
    def test_scalar_columns(self, tmp_path):
        """Test handling of scalar (1D) columns."""
        n = 500
        data = {
            "embeddings": np.random.randn(n, 768).astype(np.float32),
            "scores": np.random.randn(n).astype(np.float32),
            "ids": np.arange(n, dtype=np.int32)
        }
        
        output_file = tmp_path / "scalars.cvc"
        pack_cvc_columns(data, str(output_file))
        
        loaded = load_cvc_columns(str(output_file))
        
        assert loaded["embeddings"].shape == (n, 768)
        assert loaded["scores"].shape == (n,)
        assert loaded["ids"].shape == (n,)


class TestSectionsColumnar:
    """Tests for combined sections + columns functionality."""
    
    def test_pack_sections_columnar(self, tmp_path):
        """Test packing multi-column data from multiple sections."""
        # Create multi-modal data from different sources
        wiki_n = 100
        arxiv_n = 200
        
        wiki_data = {
            "text": np.random.randn(wiki_n, 768).astype(np.float32),
            "image": np.random.randn(wiki_n, 512).astype(np.float32),
        }
        
        arxiv_data = {
            "text": np.random.randn(arxiv_n, 768).astype(np.float32),
            "image": np.random.randn(arxiv_n, 512).astype(np.float32),
        }
        
        sections = [
            (wiki_data, {"source": "wikipedia", "date": "2024-01"}),
            (arxiv_data, {"source": "arxiv", "date": "2024-02"}),
        ]
        
        output_file = tmp_path / "sections_columnar.cvc"
        
        # Pack
        pack_cvc_sections_columnar(sections, str(output_file), chunk_size=50)
        
        assert output_file.exists()
        
        # Load all
        loaded = load_cvc_columns(str(output_file))
        
        assert loaded["text"].shape == (wiki_n + arxiv_n, 768)
        assert loaded["image"].shape == (wiki_n + arxiv_n, 512)
    
    def test_section_filtering(self, tmp_path):
        """Test filtering by section metadata."""
        wiki_n = 100
        arxiv_n = 200
        
        wiki_data = {
            "text": np.ones((wiki_n, 768), dtype=np.float32) * 1.0,
            "image": np.ones((wiki_n, 512), dtype=np.float32) * 1.0,
        }
        
        arxiv_data = {
            "text": np.ones((arxiv_n, 768), dtype=np.float32) * 2.0,
            "image": np.ones((arxiv_n, 512), dtype=np.float32) * 2.0,
        }
        
        sections = [
            (wiki_data, {"source": "wikipedia"}),
            (arxiv_data, {"source": "arxiv"}),
        ]
        
        output_file = tmp_path / "filtered.cvc"
        pack_cvc_sections_columnar(sections, str(output_file), chunk_size=50)
        
        # Load only arxiv
        loaded = load_cvc_columns(
            str(output_file),
            section_key="source",
            section_value="arxiv"
        )
        
        assert loaded["text"].shape == (arxiv_n, 768)
        assert loaded["image"].shape == (arxiv_n, 512)
        
        # Verify it's the arxiv data (all 2.0s)
        assert np.allclose(loaded["text"], 2.0, rtol=0.1)
        assert np.allclose(loaded["image"], 2.0, rtol=0.1)
    
    def test_section_and_column_filtering(self, tmp_path):
        """Test filtering by both section and column."""
        wiki_data = {
            "text": np.ones((100, 768), dtype=np.float32) * 1.0,
            "image": np.ones((100, 512), dtype=np.float32) * 1.0,
            "audio": np.ones((100, 256), dtype=np.float32) * 1.0,
        }
        
        arxiv_data = {
            "text": np.ones((200, 768), dtype=np.float32) * 2.0,
            "image": np.ones((200, 512), dtype=np.float32) * 2.0,
            "audio": np.ones((200, 256), dtype=np.float32) * 2.0,
        }
        
        sections = [
            (wiki_data, {"source": "wikipedia"}),
            (arxiv_data, {"source": "arxiv"}),
        ]
        
        output_file = tmp_path / "double_filter.cvc"
        pack_cvc_sections_columnar(sections, str(output_file), chunk_size=50)
        
        # Load only arxiv text
        loaded = load_cvc_columns(
            str(output_file),
            columns=["text"],
            section_key="source",
            section_value="arxiv"
        )
        
        assert set(loaded.keys()) == {"text"}
        assert loaded["text"].shape == (200, 768)
        assert np.allclose(loaded["text"], 2.0, rtol=0.1)


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_empty_data_error(self, tmp_path):
        """Test error on empty data dict."""
        with pytest.raises(ValueError, match="cannot be empty"):
            pack_cvc_columns({}, str(tmp_path / "test.cvc"))
    
    def test_mismatched_vector_counts(self, tmp_path):
        """Test error when columns have different vector counts."""
        data = {
            "col1": np.random.randn(100, 768).astype(np.float32),
            "col2": np.random.randn(200, 512).astype(np.float32),  # Different!
        }
        
        with pytest.raises(ValueError, match="same number of vectors"):
            pack_cvc_columns(data, str(tmp_path / "test.cvc"))
    
    def test_invalid_column_name(self, tmp_path):
        """Test error when loading non-existent column."""
        data = {
            "col1": np.random.randn(100, 768).astype(np.float32),
        }
        
        output_file = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(output_file))
        
        with pytest.raises(ValueError, match="not found"):
            load_cvc_columns(str(output_file), columns=["nonexistent"])
    
    def test_mismatched_dimensions_in_sections(self, tmp_path):
        """Test error when sections have different column dimensions."""
        wiki_data = {
            "text": np.random.randn(100, 768).astype(np.float32),
        }
        
        arxiv_data = {
            "text": np.random.randn(200, 512).astype(np.float32),  # Different dim!
        }
        
        sections = [
            (wiki_data, {"source": "wikipedia"}),
            (arxiv_data, {"source": "arxiv"}),
        ]
        
        with pytest.raises(ValueError, match="different dimensions"):
            pack_cvc_sections_columnar(sections, str(tmp_path / "test.cvc"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
