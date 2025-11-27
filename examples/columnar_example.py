"""
Example: Multi-column CVC for multi-modal embeddings.

Demonstrates the new v2.0 columnar storage feature.
"""

import numpy as np
from decompressed import (
    pack_cvc_columns,
    pack_cvc_sections_columnar,
    load_cvc_columns
)


def example_basic():
    """Basic: Store text + image + metadata together."""
    print("\n=== Basic Multi-Column Storage ===\n")
    
    n = 10_000
    data = {
        "text": np.random.randn(n, 768).astype(np.float32),
        "image": np.random.randn(n, 512).astype(np.float32),
        "doc_length": np.random.randint(10, 1000, n, dtype=np.int32),
    }
    
    pack_cvc_columns(data, "multi_modal.cvc", compressions={
        "text": "fp16",
        "image": "int8",
        "doc_length": "none"
    })
    
    loaded = load_cvc_columns("multi_modal.cvc")
    print(f"✅ Loaded {len(loaded)} columns")


def example_selective():
    """Load only specific columns (faster)."""
    print("\n=== Selective Column Loading ===\n")
    
    n = 10_000
    data = {
        "text": np.random.randn(n, 768).astype(np.float32),
        "image": np.random.randn(n, 512).astype(np.float32),
        "audio": np.random.randn(n, 256).astype(np.float32),
    }
    
    pack_cvc_columns(data, "selective.cvc")
    
    # Load only text (skip image, audio)
    text_only = load_cvc_columns("selective.cvc", columns=["text"])
    print(f"✅ Loaded 1 of 3 columns: {text_only['text'].shape}")


def example_sections():
    """Combine sections + columns."""
    print("\n=== Sections + Columns ===\n")
    
    wiki = {
        "text": np.random.randn(1_000, 768).astype(np.float32),
        "image": np.random.randn(1_000, 512).astype(np.float32),
    }
    
    arxiv = {
        "text": np.random.randn(5_000, 768).astype(np.float32),
        "image": np.random.randn(5_000, 512).astype(np.float32),
    }
    
    sections = [
        (wiki, {"source": "wikipedia"}),
        (arxiv, {"source": "arxiv"}),
    ]
    
    pack_cvc_sections_columnar(sections, "combined.cvc")
    
    # Load only arxiv text
    arxiv_text = load_cvc_columns(
        "combined.cvc",
        columns=["text"],
        section_key="source",
        section_value="arxiv"
    )
    print(f"✅ Loaded arxiv text: {arxiv_text['text'].shape}")


if __name__ == "__main__":
    example_basic()
    example_selective()
    example_sections()
    print("\n✅ All examples completed!\n")
