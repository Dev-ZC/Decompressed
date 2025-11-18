# Section-Based Packing Feature

## Overview

Added `pack_cvc_sections()` function that allows packing multiple arrays of **arbitrary sizes** into a single CVC file with section-level metadata. This solves the key limitation of chunk-based metadata where data sources must align with chunk boundaries.

## The Problem (Before)

```python
# You have:
wikipedia = np.array([...])  # 10,000 vectors
arxiv = np.array([...])      # 110,000 vectors  
github = np.array([...])     # 25,000 vectors

# Chunk metadata required sizes to match:
# - If chunk_size=100k, you'd need 100k, 100k, 100k...
# - Your sections (10k, 110k, 25k) don't align!
# - Can't easily filter by source
```

## The Solution (Now)

```python
from decompressed import pack_cvc_sections, load_cvc_range

# Pack ANY sized arrays together with metadata
sections = [
    (wikipedia, {"source": "wikipedia", "quality": "high"}),
    (arxiv, {"source": "arxiv", "quality": "high", "date": "2024-02"}),
    (github, {"source": "github", "quality": "medium"}),
]

pack_cvc_sections(sections, "combined.cvc", chunk_size=10_000)

# Load only what you need - ONE LINE!
arxiv_only = load_cvc_range("combined.cvc", 
                            section_key="source", 
                            section_value="arxiv")
# Returns exactly 110k vectors (only arXiv)
```

## Key Features

### 1. Arbitrary Section Sizes
- Pack arrays of **any size** together
- Sections don't need to align with chunk boundaries
- System automatically tracks which chunks contain which sections

### 2. Automatic Chunk Extraction
- Loads only chunks that contain the requested section
- Extracts only the relevant portion from each chunk
- Handles sections that span multiple chunks seamlessly

### 3. Rich Metadata
```python
sections = [
    (data1, {
        "source": "arxiv",
        "date": "2024-02",
        "quality": "high",
        "topic": "ml",
        "any_custom_field": "any_value"
    }),
    # ... more sections
]
```

### 4. Efficient Loading
```python
# Load by source
arxiv = load_cvc_range("file.cvc", section_key="source", section_value="arxiv")

# Load by quality
high_qual = load_cvc_range("file.cvc", section_key="quality", section_value="high")

# Load by date
recent = load_cvc_range("file.cvc", section_key="date", section_value="2024-02")

# Works with GPU too!
arxiv_gpu = load_cvc_range("file.cvc", 
                           section_key="source", 
                           section_value="arxiv",
                           device="cuda")
```

## Implementation Details

### File Format

The header now stores:
```json
{
  "sections": [
    {
      "start_index": 0,
      "end_index": 10000,
      "num_vectors": 10000,
      "metadata": {"source": "wikipedia"}
    },
    {
      "start_index": 10000,
      "end_index": 120000,
      "num_vectors": 110000,
      "metadata": {"source": "arxiv"}
    }
  ],
  "chunks": [
    {
      "rows": 10000,
      "sections": [
        {
          "metadata": {"source": "wikipedia"},
          "start_in_chunk": 0,
          "end_in_chunk": 10000
        }
      ]
    },
    {
      "rows": 10000,
      "sections": [
        {
          "metadata": {"source": "arxiv"},
          "start_in_chunk": 0,
          "end_in_chunk": 10000
        }
      ]
    }
    // ... more chunks
  ]
}
```

### Loading Algorithm

1. **Filter sections**: Find all sections matching `section_key=section_value`
2. **Find chunks**: Identify which chunks intersect with matching sections
3. **Load chunks**: Load only those chunks
4. **Extract portions**: Extract only the section-specific portion from each chunk
5. **Concatenate**: Combine all extracted portions into final array

## Use Cases

### 1. Multi-Source Embeddings
```python
# Combine embeddings from different sources
sections = [
    (wiki_embeddings, {"source": "wikipedia"}),
    (arxiv_embeddings, {"source": "arxiv"}),
    (github_embeddings, {"source": "github"}),
]

# Deploy single file
pack_cvc_sections(sections, "production.cvc")

# Use only arxiv for research queries
arxiv_vecs = load_cvc_range("production.cvc", 
                            section_key="source", 
                            section_value="arxiv")
```

### 2. Quality Filtering
```python
sections = [
    (data_v1, {"quality": "low", "version": "v1"}),
    (data_v2, {"quality": "medium", "version": "v2"}),
    (data_v3, {"quality": "high", "version": "v3"}),
]

# Use only high quality in production
high_quality = load_cvc_range("embeddings.cvc",
                              section_key="quality",
                              section_value="high")
```

### 3. Temporal Data
```python
sections = [
    (jan_data, {"month": "2024-01", "quarter": "Q1"}),
    (feb_data, {"month": "2024-02", "quarter": "Q1"}),
    (mar_data, {"month": "2024-03", "quarter": "Q1"}),
]

# Analyze specific months
feb_analysis = load_cvc_range("yearly.cvc",
                              section_key="month",
                              section_value="2024-02")
```

## Performance

- **Storage**: Single compressed file (no overhead)
- **Loading**: Only decompresses required chunks
- **Memory**: Only loads requested section data
- **I/O**: Skips irrelevant chunks entirely

Example savings:
- File: 145k vectors (10k + 110k + 25k)
- Load only arXiv (110k): **Skip 35k vectors** = ~54MB saved

## API Summary

### Packing
```python
pack_cvc_sections(
    sections: list[tuple[ndarray, dict]],
    output_path: str,
    compression: str = "fp16",
    chunk_size: int = 100000
)
```

### Loading
```python
load_cvc_range(
    path: str,
    section_key: str,      # NEW
    section_value: any,    # NEW
    device: str = "cpu",
    framework: str = "torch",
    backend: str = "auto"
)
```

## Backward Compatibility

- ✅ Existing `pack_cvc()` and `load_cvc()` functions unchanged
- ✅ Files without sections work as before
- ✅ `chunk_metadata` still supported for batch processing use cases
- ✅ All existing examples and tests continue to work

## Examples

See `examples/section_based_packing.py` for a complete working example.

## Testing

All tests pass, including:
- Packing multiple sections with arbitrary sizes
- Loading by different metadata keys
- Extracting correct portions from chunks
- FP16 compression precision validation
- Error handling for nonexistent sections
- GPU loading support
