"""Decompressed: GPU-native decompression for vector embeddings."""

__version__ = "0.1.0"

from .pycvc import load_cvc, pack_cvc, get_available_backends

__all__ = ['load_cvc', 'pack_cvc', 'get_available_backends', '__version__']