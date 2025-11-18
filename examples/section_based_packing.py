#!/usr/bin/env python3
"""
Example: Section-Based Packing and Filtering

This example demonstrates how to pack multiple arrays of different sizes
into a single CVC file with section-level metadata, then selectively load
specific sections without loading the entire file.

Use case: You have embeddings from multiple sources (Wikipedia, arXiv, GitHub)
with different sizes that don't align with chunk boundaries.
"""

import numpy as np
from decompressed import pack_cvc_sections, load_cvc_range, get_cvc_info


def main():
    print("=" * 70)
    print("Section-Based Packing and Filtering Example")
    print("=" * 70)
    
    # Step 1: Create embeddings from different sources with different sizes
    print("\n1. Creating embeddings from different sources...")
    
    wikipedia = np.random.randn(10_000, 768).astype(np.float32)
    arxiv = np.random.randn(110_000, 768).astype(np.float32)
    github = np.random.randn(25_000, 768).astype(np.float32)
    
    print(f"   Wikipedia: {wikipedia.shape[0]:,} vectors")
    print(f"   arXiv:     {arxiv.shape[0]:,} vectors")
    print(f"   GitHub:    {github.shape[0]:,} vectors")
    print(f"   Total:     {wikipedia.shape[0] + arxiv.shape[0] + github.shape[0]:,} vectors")
    
    # Step 2: Pack all sections into one file with metadata
    print("\n2. Packing all sections into one file...")
    
    sections = [
        (wikipedia, {
            "source": "wikipedia",
            "date": "2024-01",
            "quality": "high",
            "topic": "general"
        }),
        (arxiv, {
            "source": "arxiv",
            "date": "2024-02",
            "quality": "high",
            "topic": "research"
        }),
        (github, {
            "source": "github",
            "date": "2024-03",
            "quality": "medium",
            "topic": "code"
        }),
    ]
    
    pack_cvc_sections(
        sections,
        output_path="multi_source.cvc",
        compression="fp16",
        chunk_size=10_000  # Chunk size doesn't need to match section sizes!
    )
    
    print("   ✓ Created multi_source.cvc")
    
    # Step 3: Inspect the file
    print("\n3. Inspecting file structure...")
    
    info = get_cvc_info("multi_source.cvc")
    print(f"   Total vectors: {info['num_vectors']:,}")
    print(f"   Chunks: {info['num_chunks']}")
    print(f"   Dimension: {info['dimension']}")
    
    # Step 4: Load specific sections by metadata
    print("\n4. Loading sections by metadata...")
    
    # Load only arXiv (110k vectors)
    print("\n   a) Load only arXiv source:")
    arxiv_vectors = load_cvc_range(
        "multi_source.cvc",
        section_key="source",
        section_value="arxiv"
    )
    print(f"      Loaded {arxiv_vectors.shape[0]:,} vectors")
    print(f"      Memory saved: ~{(145_000 - 110_000) * 768 * 2 / 1_000_000:.1f} MB")
    
    # Load only high-quality sections (Wikipedia + arXiv = 120k)
    print("\n   b) Load all high-quality sections:")
    high_quality = load_cvc_range(
        "multi_source.cvc",
        section_key="quality",
        section_value="high"
    )
    print(f"      Loaded {high_quality.shape[0]:,} vectors (wikipedia + arxiv)")
    
    # Load by date
    print("\n   c) Load February 2024 data:")
    feb_data = load_cvc_range(
        "multi_source.cvc",
        section_key="date",
        section_value="2024-02"
    )
    print(f"      Loaded {feb_data.shape[0]:,} vectors")
    
    # Load research-related content
    print("\n   d) Load research topic:")
    research = load_cvc_range(
        "multi_source.cvc",
        section_key="topic",
        section_value="research"
    )
    print(f"      Loaded {research.shape[0]:,} vectors")
    
    # Step 5: GPU loading also works
    print("\n5. GPU loading with section filtering...")
    try:
        arxiv_gpu = load_cvc_range(
            "multi_source.cvc",
            section_key="source",
            section_value="arxiv",
            device="cuda"
        )
        print(f"   ✓ Loaded {arxiv_gpu.shape[0]:,} arXiv vectors directly to GPU")
    except Exception as e:
        print(f"   ℹ GPU not available: {e}")
    
    # Step 6: Practical use case
    print("\n6. Practical use case - Filter and process:")
    print("   Process only high-quality, recent research papers:")
    
    # In a real scenario, you might have more complex filtering needs
    # For now, we can show loading specific sections
    print(f"   → Loaded arXiv: {arxiv_vectors.shape[0]:,} vectors")
    print(f"   → Can now run similarity search, RAG, etc. on this subset")
    
    # Cleanup
    import os
    os.remove("multi_source.cvc")
    
    print("\n" + "=" * 70)
    print("Key Benefits of Section-Based Packing:")
    print("=" * 70)
    print("✓ Combine arrays of ANY size (no alignment needed)")
    print("✓ Single self-contained file for deployment")
    print("✓ Load only what you need (reduce I/O and memory)")
    print("✓ Multiple metadata fields per section")
    print("✓ Works with all devices (CPU, CUDA) and frameworks (torch, cupy)")
    print("✓ No external database required")
    

if __name__ == "__main__":
    main()
