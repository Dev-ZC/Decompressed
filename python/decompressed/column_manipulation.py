"""Column manipulation operations for multi-column CVC files."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import tempfile
import os

from .columnar_loader import ColumnarCVCLoader
from .columnar import pack_cvc_columns, pack_cvc_sections_columnar
from .schema import infer_column_schema


def add_column(
    path: str,
    column_name: str,
    data: np.ndarray,
    compression: Optional[str] = None,
    output_path: Optional[str] = None
):
    """
    Add a new column to an existing columnar CVC file.
    
    This operation rewrites the entire file with the new column included.
    For large files, this may take some time.
    
    Args:
        path: Path to existing columnar CVC file
        column_name: Name for the new column
        data: NumPy array with shape (n_vectors,) or (n_vectors, dimension)
        compression: Compression type ("fp16", "int8", "none"). 
                    Auto-inferred if None.
        output_path: Output path. If None, overwrites original file.
    
    Example:
        >>> # Add audio embeddings to existing text+image file
        >>> audio_embs = np.random.randn(1_000_000, 256).astype(np.float32)
        >>> add_column("multi_modal.cvc", "audio", audio_embs, compression="fp16")
    """
    path = Path(path)
    
    # Load existing data
    loader = ColumnarCVCLoader()
    existing_data = loader.load(str(path))
    
    # Validate new column
    num_vectors = next(iter(existing_data.values())).shape[0]
    if data.shape[0] != num_vectors:
        raise ValueError(
            f"New column must have {num_vectors} vectors, got {data.shape[0]}"
        )
    
    if column_name in existing_data:
        raise ValueError(f"Column '{column_name}' already exists. Use update_column() instead.")
    
    # Infer compression if not specified
    if compression is None:
        compression = infer_column_schema(column_name, data).compression
    
    # Add new column
    existing_data[column_name] = data
    
    # Build compressions dict
    from .pycvc import get_cvc_info
    info = get_cvc_info(str(path))
    compressions = {col["name"]: col["compression"] for col in info["columns"]}
    compressions[column_name] = compression
    
    # Determine output path
    if output_path is None:
        # Use temporary file then replace original
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cvc') as tmp:
            temp_path = tmp.name
        output_path = temp_path
        replace_original = True
    else:
        replace_original = False
    
    # Repack with new column
    pack_cvc_columns(existing_data, output_path, compressions=compressions)
    
    # Replace original if needed
    if replace_original:
        os.replace(output_path, path)
        print(f"✅ Added column '{column_name}' to {path}")
    else:
        print(f"✅ Added column '{column_name}' to {output_path}")


def update_column(
    path: str,
    column_name: str,
    data: np.ndarray,
    compression: Optional[str] = None,
    output_path: Optional[str] = None
):
    """
    Update an existing column in a columnar CVC file.
    
    This operation rewrites the entire file with updated data.
    
    Args:
        path: Path to existing columnar CVC file
        column_name: Name of column to update
        data: New data for the column
        compression: New compression type (or None to keep existing)
        output_path: Output path. If None, overwrites original file.
    
    Example:
        >>> # Update text embeddings with new model
        >>> new_text = np.random.randn(1_000_000, 768).astype(np.float32)
        >>> update_column("multi_modal.cvc", "text", new_text)
    """
    path = Path(path)
    
    # Load existing data
    loader = ColumnarCVCLoader()
    existing_data = loader.load(str(path))
    
    # Validate column exists
    if column_name not in existing_data:
        raise ValueError(
            f"Column '{column_name}' not found. Available: {list(existing_data.keys())}"
        )
    
    # Validate new data shape
    num_vectors = next(iter(existing_data.values())).shape[0]
    if data.shape[0] != num_vectors:
        raise ValueError(
            f"Updated column must have {num_vectors} vectors, got {data.shape[0]}"
        )
    
    # Update column
    existing_data[column_name] = data
    
    # Build compressions dict
    from .pycvc import get_cvc_info
    info = get_cvc_info(str(path))
    compressions = {col["name"]: col["compression"] for col in info["columns"]}
    
    # Update compression if specified
    if compression is not None:
        compressions[column_name] = compression
    
    # Determine output path
    if output_path is None:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cvc') as tmp:
            temp_path = tmp.name
        output_path = temp_path
        replace_original = True
    else:
        replace_original = False
    
    # Repack with updated column
    pack_cvc_columns(existing_data, output_path, compressions=compressions)
    
    # Replace original if needed
    if replace_original:
        os.replace(output_path, path)
        print(f"✅ Updated column '{column_name}' in {path}")
    else:
        print(f"✅ Updated column '{column_name}' in {output_path}")


def delete_column(
    path: str,
    column_name: str,
    output_path: Optional[str] = None
):
    """
    Delete a column from a columnar CVC file.
    
    This operation rewrites the file without the specified column.
    
    Args:
        path: Path to existing columnar CVC file
        column_name: Name of column to delete
        output_path: Output path. If None, overwrites original file.
    
    Example:
        >>> # Remove audio column
        >>> delete_column("multi_modal.cvc", "audio")
    """
    path = Path(path)
    
    # Load existing data
    loader = ColumnarCVCLoader()
    existing_data = loader.load(str(path))
    
    # Validate column exists
    if column_name not in existing_data:
        raise ValueError(
            f"Column '{column_name}' not found. Available: {list(existing_data.keys())}"
        )
    
    # Check if this is the last column
    if len(existing_data) == 1:
        raise ValueError(
            "Cannot delete the last column. File must have at least one column."
        )
    
    # Remove column
    del existing_data[column_name]
    
    # Build compressions dict (excluding deleted column)
    from .pycvc import get_cvc_info
    info = get_cvc_info(str(path))
    compressions = {
        col["name"]: col["compression"] 
        for col in info["columns"] 
        if col["name"] != column_name
    }
    
    # Determine output path
    if output_path is None:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cvc') as tmp:
            temp_path = tmp.name
        output_path = temp_path
        replace_original = True
    else:
        replace_original = False
    
    # Repack without deleted column
    pack_cvc_columns(existing_data, output_path, compressions=compressions)
    
    # Replace original if needed
    if replace_original:
        os.replace(output_path, path)
        print(f"✅ Deleted column '{column_name}' from {path}")
    else:
        print(f"✅ Deleted column '{column_name}', saved to {output_path}")


def rename_column(
    path: str,
    old_name: str,
    new_name: str,
    output_path: Optional[str] = None
):
    """
    Rename a column in a columnar CVC file.
    
    This operation rewrites the file with the renamed column.
    
    Args:
        path: Path to existing columnar CVC file
        old_name: Current column name
        new_name: New column name
        output_path: Output path. If None, overwrites original file.
    
    Example:
        >>> # Rename "text" to "text_embeddings"
        >>> rename_column("multi_modal.cvc", "text", "text_embeddings")
    """
    path = Path(path)
    
    # Load existing data
    loader = ColumnarCVCLoader()
    existing_data = loader.load(str(path))
    
    # Validate old column exists
    if old_name not in existing_data:
        raise ValueError(
            f"Column '{old_name}' not found. Available: {list(existing_data.keys())}"
        )
    
    # Validate new name doesn't exist
    if new_name in existing_data:
        raise ValueError(f"Column '{new_name}' already exists")
    
    # Rename column (preserve order)
    renamed_data = {}
    for key, value in existing_data.items():
        if key == old_name:
            renamed_data[new_name] = value
        else:
            renamed_data[key] = value
    
    # Build compressions dict with renamed column
    from .pycvc import get_cvc_info
    info = get_cvc_info(str(path))
    compressions = {}
    for col in info["columns"]:
        if col["name"] == old_name:
            compressions[new_name] = col["compression"]
        else:
            compressions[col["name"]] = col["compression"]
    
    # Determine output path
    if output_path is None:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cvc') as tmp:
            temp_path = tmp.name
        output_path = temp_path
        replace_original = True
    else:
        replace_original = False
    
    # Repack with renamed column
    pack_cvc_columns(renamed_data, output_path, compressions=compressions)
    
    # Replace original if needed
    if replace_original:
        os.replace(output_path, path)
        print(f"✅ Renamed column '{old_name}' → '{new_name}' in {path}")
    else:
        print(f"✅ Renamed column '{old_name}' → '{new_name}', saved to {output_path}")


def list_columns(path: str) -> List[Dict]:
    """
    List all columns in a columnar CVC file with their metadata.
    
    Args:
        path: Path to columnar CVC file
    
    Returns:
        List of column metadata dicts
    
    Example:
        >>> columns = list_columns("multi_modal.cvc")
        >>> for col in columns:
        >>>     print(f"{col['name']}: dim={col['dimension']}, compression={col['compression']}")
    """
    from .pycvc import get_cvc_info
    
    info = get_cvc_info(str(path))
    
    if info["format_type"] != "columnar":
        raise ValueError(
            f"File is not columnar format (format_type='{info['format_type']}'). "
            f"Use get_cvc_info() for single-column files."
        )
    
    return info["columns"]
