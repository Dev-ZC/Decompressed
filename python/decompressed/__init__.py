"""Decompressed: GPU-native decompression for vector embeddings."""

__version__ = "0.1.0"

from .pycvc import (
    load_cvc, 
    pack_cvc,
    pack_cvc_sections,
    get_available_backends, 
    get_backend_errors,
    get_cvc_info,
    load_cvc_chunked,
    load_cvc_range,
)
from .validation import validate_cvc, validate_cvc_integrity
from .loader import CorruptedChunkError
from .migration import upgrade_cvc, validate_cvc_version, detect_version

# Multi-column features (v2.0)
from .columnar import pack_cvc_columns, pack_cvc_sections_columnar
from .columnar_loader import ColumnarCVCLoader

# Column manipulation (v2.1)
from .column_manipulation import (
    add_column,
    update_column,
    delete_column,
    rename_column,
    list_columns
)

# Memory-mapped loader (v2.2)
from .mmap_loader import MMapCVCLoader

# Create singleton loader for columnar files
_columnar_loader = ColumnarCVCLoader()

def load_cvc_columns(path, columns=None, device="cpu", framework="torch", 
                     backend="auto", section_key=None, section_value=None):
    """Load multi-column CVC file with selective column loading."""
    return _columnar_loader.load(
        path, columns=columns, device=device, framework=framework,
        backend=backend, section_key=section_key, section_value=section_value
    )

__all__ = [
    'load_cvc', 
    'pack_cvc',
    'pack_cvc_sections',
    'get_available_backends', 
    'get_backend_errors',
    'get_cvc_info',
    'load_cvc_chunked',
    'load_cvc_range',
    # Multi-column API (v2.0)
    'pack_cvc_columns',
    'pack_cvc_sections_columnar',
    'load_cvc_columns',
    # Column manipulation (v2.1)
    'add_column',
    'update_column',
    'delete_column',
    'rename_column',
    'list_columns',
    # Memory-mapped loading (v2.2)
    'MMapCVCLoader',
    # Validation & migration
    'validate_cvc',
    'validate_cvc_integrity',
    'upgrade_cvc',
    'validate_cvc_version',
    'detect_version',
    'CorruptedChunkError',
    '__version__',
]