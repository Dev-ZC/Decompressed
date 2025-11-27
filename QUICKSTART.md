# Decompressed Quick Start Guide

**Get started with GPU-native vector compression in 5 minutes**

---

## Installation

```bash
# CPU only
pip install decompressed

# With GPU support (recommended)
pip install decompressed[gpu]
```

---

## 1. Basic Compression (Single Column)

### Pack vectors with compression

```python
import numpy as np
from decompressed import pack_cvc, load_cvc

# Your embeddings
embeddings = np.random.randn(1_000_000, 768).astype(np.float32)

# Pack with FP16 compression (2× smaller)
pack_cvc(embeddings, "embeddings.cvc", compression="fp16", chunk_size=100_000)

# Or INT8 for 4× compression (slight quality loss)
pack_cvc(embeddings, "embeddings_int8.cvc", compression="int8")
```

### Load vectors

```python
# Load to CPU
vectors = load_cvc("embeddings.cvc", device="cpu")

# Load directly to GPU (if available)
vectors_gpu = load_cvc("embeddings.cvc", device="cuda")
```

**Result**: 2-4× smaller files, GPU-native decompression

---

## 2. Multi-Column Storage

### Pack heterogeneous data

```python
from decompressed import pack_cvc_columns, load_cvc_columns

# Multi-modal embeddings
data = {
    "text": np.random.randn(1_000_000, 768).astype(np.float32),
    "image": np.random.randn(1_000_000, 512).astype(np.float32),
    "audio": np.random.randn(1_000_000, 256).astype(np.float32),
    "doc_id": np.arange(1_000_000, dtype=np.int32)
}

# Pack with per-column compression
pack_cvc_columns(data, "multi_modal.cvc", compressions={
    "text": "fp16",
    "image": "int8",
    "audio": "fp16",
    "doc_id": "none"
})
```

### Selective column loading

```python
# Load only text embeddings (10× faster than loading entire file)
text_only = load_cvc_columns("multi_modal.cvc", columns=["text"])

# Load multiple columns
text_and_image = load_cvc_columns(
    "multi_modal.cvc",
    columns=["text", "image"],
    device="cuda"  # GPU acceleration
)
```

**Result**: Load 1 of 10 columns → 10× less I/O

---

## 3. Column Manipulation

### Add, update, delete columns

```python
from decompressed import add_column, update_column, delete_column, rename_column

# Add new column to existing file
new_audio = np.random.randn(1_000_000, 256).astype(np.float32)
add_column("multi_modal.cvc", "audio", new_audio, compression="fp16")

# Update with new model
new_text = np.random.randn(1_000_000, 768).astype(np.float32)
update_column("multi_modal.cvc", "text", new_text, compression="int8")

# Remove column
delete_column("multi_modal.cvc", "audio")

# Rename column
rename_column("multi_modal.cvc", "text", "text_embeddings")
```

**Result**: Modify files without repacking everything

---

## 4. Memory-Mapped Loading

### Zero-copy access for large files

```python
from decompressed import pack_cvc, MMapCVCLoader

# Pack with page alignment for mmap
embeddings = np.random.randn(10_000_000, 768).astype(np.float32)
pack_cvc(embeddings, "large.cvc", mmap_optimized=True)

# Zero-copy loading (50-90% less memory)
with MMapCVCLoader("large.cvc") as loader:
    # File is memory-mapped, pages loaded on demand
    info = loader.get_info()
    print(f"Vectors: {info['num_vectors']:,}")
    
    # Load single chunk (zero-copy)
    chunk0 = loader.load_chunk(0)
    
    # Or load entire file (streams from mmap)
    all_vectors = loader.load()
```

**Result**: 10GB file uses ~1GB RAM (90% savings)

---

## 5. GPU-Accelerated Loading

### 20-80× faster decompression

```python
from decompressed import load_cvc_columns, get_available_backends

# Check GPU availability
backends = get_available_backends()
print(f"CUDA available: {backends['cuda']}")
print(f"Triton available: {backends['triton']}")

# Load with GPU backend (automatic)
data = load_cvc_columns(
    "multi_modal.cvc",
    columns=["text"],
    device="cuda",
    backend="auto"  # auto-selects best GPU backend
)

# Explicit backend selection
data = load_cvc_columns(
    "multi_modal.cvc",
    columns=["text"],
    device="cuda",
    backend="triton"  # or "cuda"
)
```

**Result**: 
- CPU decompression: 5 GB/s
- GPU decompression: 200 GB/s → **40× faster**

---

## 6. Inspect File Metadata

### Schema introspection

```python
from decompressed import get_cvc_info, list_columns

# Get file info (works for v1.x and v2.x)
info = get_cvc_info("multi_modal.cvc")
print(f"Format: {info['format_type']}")
print(f"Vectors: {info['num_vectors']:,}")
print(f"Columns: {len(info['columns'])}")
print(f"File size: {info['file_size'] / 1e9:.2f} GB")

# List columns with details
columns = list_columns("multi_modal.cvc")
for col in columns:
    print(f"{col['name']}: dim={col['dimension']}, compression={col['compression']}")
```

---

## 7. Sections (Multi-Source Data)

### Pack data from different sources

```python
from decompressed import pack_cvc_sections_columnar, load_cvc_columns

# Data from Wikipedia
wiki_data = {
    "text": wiki_text_embeddings,    # (100K, 768)
    "image": wiki_image_embeddings,  # (100K, 512)
}

# Data from arXiv
arxiv_data = {
    "text": arxiv_text_embeddings,   # (200K, 768)
    "image": arxiv_image_embeddings, # (200K, 512)
}

# Pack with section metadata
sections = [
    (wiki_data, {"source": "wikipedia", "date": "2024-01"}),
    (arxiv_data, {"source": "arxiv", "date": "2024-02"}),
]

pack_cvc_sections_columnar(sections, "combined.cvc")

# Load specific section
wiki_only = load_cvc_columns(
    "combined.cvc",
    section_key="source",
    section_value="wikipedia"
)
```

---

## Performance Comparison

| Feature | Traditional | Decompressed | Speedup |
|---------|------------|--------------|---------|
| File size (FP32 → FP16) | 10 GB | 5 GB | **2×** |
| File size (FP32 → INT8) | 10 GB | 2.5 GB | **4×** |
| Selective loading (1/10 cols) | Load all | Load 1 | **10×** |
| GPU decompression | 5 GB/s | 200 GB/s | **40×** |
| Memory (10GB file, mmap) | 10 GB | 1 GB | **10×** |

---

## Common Patterns

### Pattern 1: Multi-Modal Search Pipeline

```python
# Initial search with text only
text_embs = load_cvc_columns("data.cvc", columns=["text"], device="cuda")
results = vector_search(query, text_embs)

# Rerank with images
image_embs = load_cvc_columns("data.cvc", columns=["image"], device="cuda")
final_results = rerank_with_images(results, image_embs)
```

### Pattern 2: Incremental Updates

```python
# Add new data without repacking entire file
new_vectors = np.random.randn(10_000, 768).astype(np.float32)
add_column("embeddings.cvc", "new_model_v2", new_vectors)

# Compare old vs new
old = load_cvc_columns("embeddings.cvc", columns=["old_model"])
new = load_cvc_columns("embeddings.cvc", columns=["new_model_v2"])
```

### Pattern 3: Large File Processing

```python
# Stream process 100GB file with minimal memory
with MMapCVCLoader("huge.cvc") as loader:
    for chunk_idx in range(loader.header["num_chunks"]):
        chunk = loader.load_chunk(chunk_idx, device="cuda")
        process_on_gpu(chunk)  # Only 1 chunk in memory at a time
```

---

## Next Steps

- **[README.md](README.md)** - Full documentation and API reference
- **[format.md](format.md)** - File format specification
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and updates

**Questions?** Open an issue on GitHub!
