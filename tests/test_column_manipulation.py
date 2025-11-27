"""Tests for column manipulation operations."""

import numpy as np
import pytest
from pathlib import Path

from decompressed import (
    pack_cvc_columns,
    load_cvc_columns,
    get_cvc_info
)
from decompressed.column_manipulation import (
    add_column,
    update_column,
    delete_column,
    rename_column,
    list_columns
)


class TestAddColumn:
    """Test adding columns to existing files."""
    
    def test_add_column_basic(self, tmp_path):
        """Test basic column addition."""
        # Create initial file with 2 columns
        n = 1000
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Add audio column
        audio_data = np.random.randn(n, 256).astype(np.float32)
        add_column(str(file_path), "audio", audio_data, compression="fp16")
        
        # Verify
        loaded = load_cvc_columns(str(file_path))
        assert set(loaded.keys()) == {"text", "image", "audio"}
        assert loaded["audio"].shape == (n, 256)
        # FP16 compression has ~0.005 relative error
        np.testing.assert_allclose(loaded["audio"], audio_data, rtol=0.01, atol=0.01)
    
    def test_add_column_scalar(self, tmp_path):
        """Test adding scalar column."""
        n = 500
        initial_data = {
            "embeddings": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Add scalar column
        doc_ids = np.arange(n, dtype=np.int32)
        add_column(str(file_path), "doc_id", doc_ids, compression="none")
        
        # Verify
        loaded = load_cvc_columns(str(file_path))
        assert "doc_id" in loaded
        assert loaded["doc_id"].shape == (n,)
    
    def test_add_column_with_output_path(self, tmp_path):
        """Test adding column to new file."""
        n = 500
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
        }
        
        input_file = tmp_path / "input.cvc"
        output_file = tmp_path / "output.cvc"
        
        pack_cvc_columns(initial_data, str(input_file))
        
        # Add column to new file
        new_col = np.random.randn(n, 512).astype(np.float32)
        add_column(str(input_file), "image", new_col, output_path=str(output_file))
        
        # Verify original unchanged
        original = load_cvc_columns(str(input_file))
        assert set(original.keys()) == {"text"}
        
        # Verify new file has both columns
        updated = load_cvc_columns(str(output_file))
        assert set(updated.keys()) == {"text", "image"}
    
    def test_add_column_wrong_size_error(self, tmp_path):
        """Test error when adding column with wrong size."""
        n = 1000
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Try to add column with wrong size
        wrong_size = np.random.randn(500, 512).astype(np.float32)
        
        with pytest.raises(ValueError, match="must have 1000 vectors"):
            add_column(str(file_path), "image", wrong_size)
    
    def test_add_duplicate_column_error(self, tmp_path):
        """Test error when adding column that already exists."""
        n = 500
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Try to add duplicate
        duplicate = np.random.randn(n, 768).astype(np.float32)
        
        with pytest.raises(ValueError, match="already exists"):
            add_column(str(file_path), "text", duplicate)


class TestUpdateColumn:
    """Test updating existing columns."""
    
    def test_update_column_basic(self, tmp_path):
        """Test basic column update."""
        n = 1000
        initial_data = {
            "text": np.ones((n, 768), dtype=np.float32) * 1.0,
            "image": np.ones((n, 512), dtype=np.float32) * 2.0,
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Update text column
        new_text = np.ones((n, 768), dtype=np.float32) * 5.0
        update_column(str(file_path), "text", new_text)
        
        # Verify
        loaded = load_cvc_columns(str(file_path))
        assert np.allclose(loaded["text"], 5.0, rtol=1e-3)
        assert np.allclose(loaded["image"], 2.0, rtol=0.1)  # Unchanged
    
    def test_update_column_change_compression(self, tmp_path):
        """Test updating column with different compression."""
        n = 500
        initial_data = {
            "embeddings": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path), compressions={"embeddings": "fp16"})
        
        # Update with int8 compression
        new_data = np.random.randn(n, 768).astype(np.float32)
        update_column(str(file_path), "embeddings", new_data, compression="int8")
        
        # Verify compression changed
        info = get_cvc_info(str(file_path))
        assert info["columns"][0]["compression"] == "int8"
    
    def test_update_nonexistent_column_error(self, tmp_path):
        """Test error when updating non-existent column."""
        n = 500
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Try to update non-existent column
        fake_data = np.random.randn(n, 512).astype(np.float32)
        
        with pytest.raises(ValueError, match="not found"):
            update_column(str(file_path), "image", fake_data)


class TestDeleteColumn:
    """Test deleting columns."""
    
    def test_delete_column_basic(self, tmp_path):
        """Test basic column deletion."""
        n = 1000
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
            "audio": np.random.randn(n, 256).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Delete audio column
        delete_column(str(file_path), "audio")
        
        # Verify
        loaded = load_cvc_columns(str(file_path))
        assert set(loaded.keys()) == {"text", "image"}
        assert "audio" not in loaded
    
    def test_delete_last_column_error(self, tmp_path):
        """Test error when deleting the last column."""
        n = 500
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Try to delete last column
        with pytest.raises(ValueError, match="last column"):
            delete_column(str(file_path), "text")
    
    def test_delete_nonexistent_column_error(self, tmp_path):
        """Test error when deleting non-existent column."""
        n = 500
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        with pytest.raises(ValueError, match="not found"):
            delete_column(str(file_path), "audio")


class TestRenameColumn:
    """Test renaming columns."""
    
    def test_rename_column_basic(self, tmp_path):
        """Test basic column rename."""
        n = 1000
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Rename text → text_embeddings
        rename_column(str(file_path), "text", "text_embeddings")
        
        # Verify
        loaded = load_cvc_columns(str(file_path))
        assert set(loaded.keys()) == {"text_embeddings", "image"}
        assert "text" not in loaded
    
    def test_rename_preserves_data(self, tmp_path):
        """Test that renaming preserves data."""
        n = 500
        text_data = np.random.randn(n, 768).astype(np.float32)
        initial_data = {
            "text": text_data,
            "image": np.random.randn(n, 512).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Rename
        rename_column(str(file_path), "text", "text_v2")
        
        # Verify data unchanged (FP16 tolerance)
        loaded = load_cvc_columns(str(file_path))
        np.testing.assert_allclose(loaded["text_v2"], text_data, rtol=0.01, atol=0.01)
    
    def test_rename_to_existing_name_error(self, tmp_path):
        """Test error when renaming to existing column name."""
        n = 500
        initial_data = {
            "text": np.random.randn(n, 768).astype(np.float32),
            "image": np.random.randn(n, 512).astype(np.float32),
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # Try to rename text → image (already exists)
        with pytest.raises(ValueError, match="already exists"):
            rename_column(str(file_path), "text", "image")


class TestListColumns:
    """Test listing columns."""
    
    def test_list_columns_basic(self, tmp_path):
        """Test listing columns."""
        n = 1000
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
        
        # List columns
        columns = list_columns(str(file_path))
        
        assert len(columns) == 3
        
        # Check text column
        text_col = next(c for c in columns if c["name"] == "text")
        assert text_col["dimension"] == 768
        assert text_col["compression"] == "fp16"
        
        # Check image column
        image_col = next(c for c in columns if c["name"] == "image")
        assert image_col["dimension"] == 512
        assert image_col["compression"] == "int8"
        
        # Check doc_id column
        doc_id_col = next(c for c in columns if c["name"] == "doc_id")
        assert doc_id_col["dimension"] == 1
        assert doc_id_col["compression"] == "none"


class TestIntegration:
    """Integration tests for multiple operations."""
    
    def test_multiple_operations_sequence(self, tmp_path):
        """Test sequence of operations."""
        n = 500
        
        # 1. Create file with 2 columns
        initial_data = {
            "text": np.ones((n, 768), dtype=np.float32) * 1.0,
            "image": np.ones((n, 512), dtype=np.float32) * 2.0,
        }
        
        file_path = tmp_path / "test.cvc"
        pack_cvc_columns(initial_data, str(file_path))
        
        # 2. Add audio column
        audio = np.ones((n, 256), dtype=np.float32) * 3.0
        add_column(str(file_path), "audio", audio)
        
        # 3. Rename image → image_emb
        rename_column(str(file_path), "image", "image_emb")
        
        # 4. Update text column
        new_text = np.ones((n, 768), dtype=np.float32) * 5.0
        update_column(str(file_path), "text", new_text)
        
        # 5. Delete audio column
        delete_column(str(file_path), "audio")
        
        # Verify final state
        loaded = load_cvc_columns(str(file_path))
        assert set(loaded.keys()) == {"text", "image_emb"}
        assert np.allclose(loaded["text"], 5.0, rtol=1e-3)
        assert np.allclose(loaded["image_emb"], 2.0, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
