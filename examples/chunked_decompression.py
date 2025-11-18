"""
Example: Chunked Decompression

This example demonstrates the new chunked decompression APIs in Decompressed.
Chunked decompression allows you to:
1. Load only specific chunks from a file
2. Process large files that don't fit in memory
3. Stream embeddings for incremental processing
"""

import numpy as np
from decompressed import (
    pack_cvc,
    get_cvc_info,
    load_cvc_chunked,
    load_cvc_range,
    load_cvc,
)


def example_create_sample_file():
    """Create a sample .cvc file for demonstration."""
    print("Creating sample embeddings file...")
    
    # Generate sample embeddings: 500k vectors, 768 dimensions
    num_vectors = 500_000
    dimension = 768
    embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
    
    # Pack into .cvc format with 100k vectors per chunk (5 chunks total)
    pack_cvc(
        embeddings,
        output_path="sample_embeddings.cvc",
        compression="fp16",
        chunk_size=100_000,
    )
    print(f"Created sample_embeddings.cvc: {num_vectors} vectors, {dimension} dims\n")
    return "sample_embeddings.cvc"


def example_create_file_with_metadata():
    """Create a .cvc file with custom chunk metadata."""
    print("=== Creating file with custom chunk metadata ===")
    
    # Generate sample embeddings: 250k vectors, 384 dimensions
    num_vectors = 250_000
    dimension = 384
    embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
    
    # Define metadata for each chunk (5 chunks of 50k vectors each)
    chunk_metadata = [
        {"source": "wikipedia", "date": "2024-01", "topic": "science"},
        {"source": "arxiv", "date": "2024-02", "topic": "ML"},
        {"source": "github", "date": "2024-03", "topic": "code"},
        {"source": "books", "date": "2024-04", "topic": "history"},
        {"source": "news", "date": "2024-05", "topic": "current_events"},
    ]
    
    # Pack with metadata
    pack_cvc(
        embeddings,
        output_path="sample_with_metadata.cvc",
        compression="fp16",
        chunk_size=50_000,
        chunk_metadata=chunk_metadata,
    )
    print(f"Created sample_with_metadata.cvc with custom metadata\n")
    return "sample_with_metadata.cvc"


def example_inspect_file(path):
    """Demonstrate get_cvc_info() for inspecting file metadata."""
    print("=== Example 1: Inspecting File Metadata ===")
    
    info = get_cvc_info(path)
    print(f"File: {path}")
    print(f"  Total vectors: {info['num_vectors']:,}")
    print(f"  Dimension: {info['dimension']}")
    print(f"  Compression: {info['compression']}")
    print(f"  Number of chunks: {info['num_chunks']}")
    print(f"  Chunk sizes: {[chunk['rows'] for chunk in info['chunks']]}")
    
    # Show metadata if present
    has_metadata = any(chunk['metadata'] is not None for chunk in info['chunks'])
    if has_metadata:
        print(f"  Chunk metadata:")
        for chunk in info['chunks']:
            if chunk['metadata']:
                print(f"    Chunk {chunk['index']}: {chunk['metadata']}")
    print()


def example_iterate_all_chunks(path):
    """Demonstrate load_cvc_chunked() to iterate through all chunks."""
    print("=== Example 2: Iterating Through All Chunks ===")
    
    for chunk_idx, vectors in load_cvc_chunked(path, device="cpu"):
        print(f"Processing chunk {chunk_idx}: shape={vectors.shape}, "
              f"mean={vectors.mean():.4f}, std={vectors.std():.4f}")
    print()


def example_load_specific_chunks(path):
    """Demonstrate loading only specific chunks."""
    print("=== Example 3: Loading Specific Chunks ===")
    
    # Load only chunks 0, 2, and 4
    chunk_indices = [0, 2, 4]
    print(f"Loading chunks: {chunk_indices}")
    
    for chunk_idx, vectors in load_cvc_chunked(
        path,
        chunk_indices=chunk_indices,
        device="cpu"
    ):
        print(f"  Chunk {chunk_idx}: {vectors.shape}")
    print()


def example_load_chunk_range(path):
    """Demonstrate load_cvc_range() to load and concatenate chunks."""
    print("=== Example 4: Loading Chunk Range (Concatenated) ===")
    
    # Load first 2 chunks and concatenate them
    vectors = load_cvc_range(path, chunk_indices=[0, 1], device="cpu")
    print(f"Loaded and concatenated chunks [0, 1]: shape={vectors.shape}")
    print()


def example_memory_efficient_processing(path):
    """Demonstrate memory-efficient processing of large files."""
    print("=== Example 5: Memory-Efficient Processing ===")
    
    # Process chunks one at a time without loading entire file
    chunk_stats = []
    for chunk_idx, vectors in load_cvc_chunked(path, device="cpu"):
        # Compute statistics for this chunk
        stats = {
            'chunk': chunk_idx,
            'size': vectors.shape[0],
            'mean': float(vectors.mean()),
            'std': float(vectors.std()),
            'min': float(vectors.min()),
            'max': float(vectors.max()),
        }
        chunk_stats.append(stats)
        # In a real application, you might:
        # - Compute similarities against a query
        # - Update an index incrementally
        # - Transform embeddings and write to another file
        # - etc.
    
    print("Chunk statistics computed:")
    for stats in chunk_stats:
        print(f"  Chunk {stats['chunk']}: mean={stats['mean']:.4f}, "
              f"std={stats['std']:.4f}")
    print()


def example_compare_with_full_load(path):
    """Compare chunked loading vs full file loading."""
    print("=== Example 6: Comparison with Full File Loading ===")
    
    # Method 1: Load entire file at once (original API)
    print("Loading entire file at once...")
    vectors_full = load_cvc(path, device="cpu")
    print(f"  Full load: shape={vectors_full.shape}")
    
    # Method 2: Load all chunks and concatenate (chunked API)
    print("Loading via chunked API and concatenating...")
    chunks = []
    for chunk_idx, chunk_vectors in load_cvc_chunked(path, device="cpu"):
        chunks.append(chunk_vectors)
    vectors_chunked = np.concatenate(chunks, axis=0)
    print(f"  Chunked load: shape={vectors_chunked.shape}")
    
    # Verify they're identical
    are_equal = np.allclose(vectors_full, vectors_chunked)
    print(f"  Results match: {are_equal}")
    print()


def example_metadata_filtering(path):
    """Demonstrate metadata-based chunk filtering."""
    print("=== Example 7: Metadata-Based Filtering ===")
    
    # Load only chunks from a specific source
    print("Loading chunks with source='arxiv'...")
    arxiv_vectors = load_cvc_range(path, 
                                   metadata_key="source", 
                                   metadata_value="arxiv")
    print(f"  Loaded shape: {arxiv_vectors.shape}")
    
    # Load chunks by topic
    print("\nLoading chunks with topic='ML'...")
    ml_vectors = load_cvc_range(path,
                                metadata_key="topic",
                                metadata_value="ML")
    print(f"  Loaded shape: {ml_vectors.shape}")
    
    # Compare with manual filtering
    print("\nComparing with manual filtering...")
    info = get_cvc_info(path)
    manual_indices = [
        chunk['index'] 
        for chunk in info['chunks']
        if chunk.get('metadata') and chunk['metadata'].get('source') == 'arxiv'
    ]
    manual_vectors = load_cvc_range(path, chunk_indices=manual_indices)
    are_equal = np.allclose(arxiv_vectors, manual_vectors)
    print(f"  Results match manual filtering: {are_equal}")
    print()


def main():
    """Run all examples."""
    print("Chunked Decompression Examples\n" + "="*50 + "\n")
    
    # Create sample files
    path = example_create_sample_file()
    path_with_metadata = example_create_file_with_metadata()
    
    # Run examples
    example_inspect_file(path)
    example_inspect_file(path_with_metadata)  # Show file with metadata
    example_iterate_all_chunks(path)
    example_load_specific_chunks(path)
    example_load_chunk_range(path)
    example_memory_efficient_processing(path)
    example_compare_with_full_load(path)
    example_metadata_filtering(path_with_metadata)  # New metadata filtering example
    
    print("All examples completed!")


if __name__ == "__main__":
    main()
