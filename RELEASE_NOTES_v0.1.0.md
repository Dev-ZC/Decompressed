# Release Notes - v0.1.0

Initial public release of Decompressed.

## What's New

### Section-Based Packing

Added `pack_cvc_sections()` for combining multiple arrays of different sizes into a single file with metadata. This solves the main limitation where chunk metadata required arrays to align with chunk boundaries.

```python
sections = [
    (wikipedia_embeddings, {"source": "wikipedia"}),
    (arxiv_embeddings, {"source": "arxiv"}),
]
pack_cvc_sections(sections, "combined.cvc", chunk_size=10_000)

# Load just arXiv
vectors = load_cvc_range("combined.cvc", section_key="source", section_value="arxiv")
```

The loader automatically handles chunks that span section boundaries and extracts only the requested data.

### Metadata Filtering

Extended `load_cvc_range()` to accept `section_key` and `section_value` parameters for filtering. You can now load specific sections without writing custom filtering logic.

### Core Features

- FP16 and INT8 compression with 2-4× size reduction
- CPU-based decompression (GPU backends via Triton in development)
- Chunked file format for streaming large datasets
- Per-chunk and per-section metadata support

## Installation

```bash
pip install decompressed
```

For GPU support (once available):
```bash
pip install decompressed[gpu]
```

## Known Issues

- CUDA backend not yet implemented (planned)
- Triton backend available but requires manual installation
- No pre-built wheels yet - package builds from source

## Documentation

See README.md for API documentation and examples.

## Breaking Changes

None (initial release).
