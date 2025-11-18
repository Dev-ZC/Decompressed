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

__all__ = [
    'load_cvc', 
    'pack_cvc',
    'pack_cvc_sections',
    'get_available_backends', 
    'get_backend_errors',
    'get_cvc_info',
    'load_cvc_chunked',
    'load_cvc_range',
    '__version__',
]