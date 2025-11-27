"""Tests for schema introspection via get_cvc_info()."""

import numpy as np
import pytest
from pathlib import Path

from decompressed import (
    pack_cvc,
    pack_cvc_columns,
    pack_cvc_sections_columnar,
    get_cvc_info
)


class TestSchemaIntrospection:
    """Test get_cvc_info() for both v1.x and v2.x files."""
    
    def test_get_info_single_column_v1(self, tmp_path):
        """Test get_cvc_info() for v1.x single-column files."""
        # Create v1.x file
        vectors = np.random.randn(1000, 768).astype(np.float32)
        output_file = tmp_path / "single.cvc"
        pack_cvc(vectors, str(output_file), compression="fp16", chunk_size=100)
        
        # Get info
        info = get_cvc_info(str(output_file))
        
        # Verify format
        assert info["format_type"] == "single"
        assert info["num_vectors"] == 1000
        assert info["dimension"] == 768
        assert info["compression"] == "fp16"
        assert info["num_chunks"] == 10
        assert info["file_size"] > 0
        assert "columns" not in info  # Single-column format
        
        # Verify chunks
        assert len(info["chunks"]) == 10
        for i, chunk in enumerate(info["chunks"]):
            assert chunk["index"] == i
            assert chunk["rows"] == 100
    
    def test_get_info_columnar_v2(self, tmp_path):
        """Test get_cvc_info() for v2.x columnar files."""
        # Create v2.x columnar file
        data = {
            "text": np.random.randn(1000, 768).astype(np.float32),
            "image": np.random.randn(1000, 512).astype(np.float32),
            "doc_id": np.arange(1000, dtype=np.int32)
        }
        
        output_file = tmp_path / "columnar.cvc"
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
        
        # Get info
        info = get_cvc_info(str(output_file))
        
        # Verify format
        assert info["format_type"] == "columnar"
        assert info["num_vectors"] == 1000
        assert info["num_chunks"] == 10
        assert info["file_size"] > 0
        assert "dimension" not in info  # Columnar format
        assert "compression" not in info  # Per-column compression
        
        # Verify columns
        assert "columns" in info
        columns = info["columns"]
        assert len(columns) == 3
        
        # Check column details
        text_col = next(c for c in columns if c["name"] == "text")
        assert text_col["dimension"] == 768
        assert text_col["dtype"] == "float32"
        assert text_col["compression"] == "fp16"
        
        image_col = next(c for c in columns if c["name"] == "image")
        assert image_col["dimension"] == 512
        assert image_col["compression"] == "int8"
        
        doc_id_col = next(c for c in columns if c["name"] == "doc_id")
        assert doc_id_col["dimension"] == 1  # Scalars have dimension 1
        assert doc_id_col["compression"] == "none"
        
        # Verify chunks
        assert len(info["chunks"]) == 10
        for i, chunk in enumerate(info["chunks"]):
            assert chunk["index"] == i
            assert chunk["rows"] == 100
            assert set(chunk["columns"]) == {"text", "image", "doc_id"}
    
    def test_get_info_sections_columnar(self, tmp_path):
        """Test get_cvc_info() for columnar files with sections."""
        # Create columnar file with sections
        wiki_data = {
            "text": np.random.randn(100, 768).astype(np.float32),
            "image": np.random.randn(100, 512).astype(np.float32),
        }
        
        arxiv_data = {
            "text": np.random.randn(200, 768).astype(np.float32),
            "image": np.random.randn(200, 512).astype(np.float32),
        }
        
        sections = [
            (wiki_data, {"source": "wikipedia", "date": "2024-01"}),
            (arxiv_data, {"source": "arxiv", "date": "2024-02"}),
        ]
        
        output_file = tmp_path / "sections_columnar.cvc"
        pack_cvc_sections_columnar(sections, str(output_file), chunk_size=50)
        
        # Get info
        info = get_cvc_info(str(output_file))
        
        # Verify format
        assert info["format_type"] == "columnar"
        assert info["num_vectors"] == 300
        
        # Verify columns
        assert len(info["columns"]) == 2
        column_names = {c["name"] for c in info["columns"]}
        assert column_names == {"text", "image"}
        
        # Verify sections
        assert "sections" in info
        sections_info = info["sections"]
        assert len(sections_info) == 2
        
        wiki_section = sections_info[0]
        assert wiki_section["num_vectors"] == 100
        assert wiki_section["metadata"]["source"] == "wikipedia"
        
        arxiv_section = sections_info[1]
        assert arxiv_section["num_vectors"] == 200
        assert arxiv_section["metadata"]["source"] == "arxiv"
    
    def test_get_info_file_size(self, tmp_path):
        """Test that file_size is reported correctly."""
        vectors = np.random.randn(100, 128).astype(np.float32)
        output_file = tmp_path / "test.cvc"
        pack_cvc(vectors, str(output_file), compression="fp16")
        
        # Get info
        info = get_cvc_info(str(output_file))
        
        # Verify file size matches actual file
        actual_size = output_file.stat().st_size
        assert info["file_size"] == actual_size
        assert info["file_size"] > 0


class TestSchemaDisplay:
    """Test pretty-printing of schema information."""
    
    def test_display_single_column_schema(self, tmp_path):
        """Test displaying schema for single-column files."""
        vectors = np.random.randn(1000, 768).astype(np.float32)
        output_file = tmp_path / "test.cvc"
        pack_cvc(vectors, str(output_file), compression="fp16")
        
        info = get_cvc_info(str(output_file))
        
        # Should be able to print without error
        print(f"\n=== Single-Column File ===")
        print(f"Format: {info['format_type']}")
        print(f"Vectors: {info['num_vectors']:,}")
        print(f"Dimension: {info['dimension']}")
        print(f"Compression: {info['compression']}")
        print(f"Chunks: {info['num_chunks']}")
        print(f"File size: {info['file_size'] / 1e6:.2f} MB")
    
    def test_display_columnar_schema(self, tmp_path):
        """Test displaying schema for columnar files."""
        data = {
            "text": np.random.randn(1000, 768).astype(np.float32),
            "image": np.random.randn(1000, 512).astype(np.float32),
            "doc_id": np.arange(1000, dtype=np.int32)
        }
        
        output_file = tmp_path / "test.cvc"
        pack_cvc_columns(data, str(output_file))
        
        info = get_cvc_info(str(output_file))
        
        # Should be able to print without error
        print(f"\n=== Columnar File ===")
        print(f"Format: {info['format_type']}")
        print(f"Vectors: {info['num_vectors']:,}")
        print(f"Columns: {len(info['columns'])}")
        for col in info["columns"]:
            print(f"  - {col['name']}: dim={col['dimension']}, "
                  f"dtype={col['dtype']}, compression={col['compression']}")
        print(f"Chunks: {info['num_chunks']}")
        print(f"File size: {info['file_size'] / 1e6:.2f} MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
