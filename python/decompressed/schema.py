"""Column schema definitions for multi-column CVC files."""

from typing import List, Literal, Optional
from dataclasses import dataclass, asdict


@dataclass
class ColumnSchema:
    """
    Schema for a single column in a columnar CVC file.
    
    Attributes:
        name: Column identifier (e.g., "text", "image")
        dimension: Number of dimensions (1 for scalars, N for vectors)
        dtype: NumPy dtype name
        compression: Compression scheme for this column
    """
    name: str
    dimension: int
    dtype: str
    compression: Literal["fp16", "int8", "none"]
    
    def validate(self):
        """Validate column schema."""
        if self.dimension < 1:
            raise ValueError(f"Column '{self.name}': dimension must be >= 1, got {self.dimension}")
        
        if not self.name or self.name.strip() == "":
            raise ValueError("Column name cannot be empty")
        
        # Validate dtype is recognized
        valid_dtypes = ["float32", "float16", "int32", "int64", "int8", "uint8"]
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Column '{self.name}': dtype must be one of {valid_dtypes}, got '{self.dtype}'"
            )
        
        # Validate compression
        valid_compressions = ["fp16", "int8", "none"]
        if self.compression not in valid_compressions:
            raise ValueError(
                f"Column '{self.name}': compression must be one of {valid_compressions}, "
                f"got '{self.compression}'"
            )
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "compression": self.compression
        }


@dataclass
class ColumnarFileSchema:
    """
    Complete schema for a multi-column CVC file.
    
    Attributes:
        format_type: "single" or "columnar"
        num_vectors: Total number of vectors/rows
        columns: List of column schemas
        sections: Optional section metadata (for pack_cvc_sections_columnar)
    """
    format_type: Literal["single", "columnar"]
    num_vectors: int
    columns: List[ColumnSchema]
    sections: Optional[List[dict]] = None
    
    def validate(self):
        """Validate complete schema."""
        if self.num_vectors < 1:
            raise ValueError(f"num_vectors must be >= 1, got {self.num_vectors}")
        
        if not self.columns:
            raise ValueError("Must have at least one column")
        
        # Check for duplicate column names
        names = [col.name for col in self.columns]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate column names: {set(duplicates)}")
        
        # Validate each column
        for col in self.columns:
            col.validate()
        
        return True
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "format_type": self.format_type,
            "num_vectors": self.num_vectors,
            "columns": [col.to_dict() for col in self.columns]
        }
        
        if self.sections is not None:
            result["sections"] = self.sections
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "ColumnarFileSchema":
        """Create schema from dict (loaded from JSON header)."""
        columns = [
            ColumnSchema(
                name=col["name"],
                dimension=col["dimension"],
                dtype=col["dtype"],
                compression=col["compression"]
            )
            for col in data["columns"]
        ]
        
        return cls(
            format_type=data["format_type"],
            num_vectors=data["num_vectors"],
            columns=columns,
            sections=data.get("sections")
        )


def infer_column_schema(
    name: str,
    arr,
    compression: Optional[str] = None
) -> ColumnSchema:
    """
    Infer column schema from numpy array.
    
    Args:
        name: Column name
        arr: NumPy array
        compression: Override compression (or auto-infer)
    
    Returns:
        ColumnSchema instance
    """
    import numpy as np
    
    # Determine dimension
    dimension = 1 if arr.ndim == 1 else arr.shape[1]
    
    # Determine dtype
    dtype = str(arr.dtype)
    
    # Determine compression
    if compression is None:
        # Auto-infer based on dtype
        if "float" in dtype:
            compression = "fp16"  # Default for floats
        elif "int" in dtype or "uint" in dtype:
            compression = "none"  # Don't compress integers
        else:
            compression = "none"
    
    return ColumnSchema(
        name=name,
        dimension=dimension,
        dtype=dtype,
        compression=compression
    )


def validate_column_data(columns: List[ColumnSchema], data: dict):
    """
    Validate that data dict matches column schemas.
    
    Args:
        columns: List of column schemas
        data: Dict mapping column names to arrays
    
    Raises:
        ValueError: If validation fails
    """
    import numpy as np
    
    # Check all columns present
    schema_names = {col.name for col in columns}
    data_names = set(data.keys())
    
    if schema_names != data_names:
        missing = schema_names - data_names
        extra = data_names - schema_names
        msg = []
        if missing:
            msg.append(f"Missing columns: {missing}")
        if extra:
            msg.append(f"Extra columns: {extra}")
        raise ValueError(". ".join(msg))
    
    # Check each column
    for col in columns:
        arr = data[col.name]
        
        # Check dimension
        expected_dim = col.dimension
        actual_dim = 1 if arr.ndim == 1 else arr.shape[1]
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Column '{col.name}': expected dimension {expected_dim}, "
                f"got {actual_dim}"
            )
        
        # Check dtype compatibility
        if str(arr.dtype) != col.dtype:
            raise ValueError(
                f"Column '{col.name}': expected dtype {col.dtype}, "
                f"got {arr.dtype}"
            )
