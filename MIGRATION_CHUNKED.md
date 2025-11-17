# Migration Guide: Adopting Chunked Decompression

This guide helps existing Decompressed users adopt the new chunked decompression APIs for improved memory efficiency and performance.

## No Breaking Changes

**Good news:** All existing code continues to work without modifications. The chunked decompression APIs are purely additive.

## When to Use Chunked Decompression

Consider using chunked decompression if:

- ✅ Your `.cvc` files are larger than available RAM
- ✅ You need to process embeddings incrementally
- ✅ You only need a subset of vectors from a large file
- ✅ You want faster time-to-first-result
- ✅ You're distributing processing across multiple GPUs/machines
- ✅ You're implementing streaming or online algorithms

Stick with `load_cvc()` if:

- ✅ Your files fit comfortably in memory
- ✅ You need all vectors at once
- ✅ You prefer simplicity over memory optimization

## Migration Examples

### Example 1: Memory-Efficient Processing

**Before (loads entire file):**

```python
from decompressed import load_cvc

# This loads all 10M vectors at once (30GB in FP32)
vectors = load_cvc("large_embeddings.cvc", device="cpu")

# Process all vectors
for i in range(len(vectors)):
    result = process_vector(vectors[i])
    save_result(result)
```

**After (processes chunks):**

```python
from decompressed import load_cvc_chunked

# This loads one chunk at a time (only ~300MB in memory)
for chunk_idx, chunk_vectors in load_cvc_chunked("large_embeddings.cvc", device="cpu"):
    # Process this chunk
    for vector in chunk_vectors:
        result = process_vector(vector)
        save_result(result)
    # Memory is freed before loading next chunk
```

**Benefit:** Reduces memory usage from 30GB to ~300MB.

---

### Example 2: Early Exit / Search

**Before (loads everything first):**

```python
from decompressed import load_cvc

# Load entire file even though we might exit early
vectors = load_cvc("embeddings.cvc", device="cuda")

# Search for nearest neighbor
for i, vec in enumerate(vectors):
    similarity = compute_similarity(query, vec)
    if similarity > threshold:
        print(f"Found match at index {i}")
        break  # But we already loaded everything!
```

**After (loads on-demand):**

```python
from decompressed import load_cvc_chunked

# Load chunks one at a time
offset = 0
for chunk_idx, chunk_vectors in load_cvc_chunked("embeddings.cvc", device="cuda"):
    for i, vec in enumerate(chunk_vectors):
        similarity = compute_similarity(query, vec)
        if similarity > threshold:
            print(f"Found match at index {offset + i}")
            break  # Only loaded the chunks we needed!
    offset += len(chunk_vectors)
```

**Benefit:** Faster time-to-result, loads only necessary data.

---

### Example 3: Partial File Access

**Before (loads entire file):**

```python
from decompressed import load_cvc

# Load everything even though we only need 10%
all_vectors = load_cvc("embeddings.cvc", device="cpu")

# Use only first 10%
subset = all_vectors[:100000]
process(subset)
```

**After (loads only what's needed):**

```python
from decompressed import get_cvc_info, load_cvc_range

# Check how many chunks we need
info = get_cvc_info("embeddings.cvc")
vectors_per_chunk = info['chunks'][0]['rows']
chunks_needed = 100000 // vectors_per_chunk

# Load only the first few chunks
subset = load_cvc_range("embeddings.cvc", 
                       chunk_indices=list(range(chunks_needed)), 
                       device="cpu")
process(subset)
```

**Benefit:** 90% reduction in I/O and memory usage.

---

### Example 4: Distributed Processing

**Before (single machine):**

```python
from decompressed import load_cvc

# Single machine loads and processes everything
vectors = load_cvc("embeddings.cvc", device="cuda:0")
results = process_all(vectors)
```

**After (distributed across GPUs):**

```python
from decompressed import get_cvc_info, load_cvc_range

# Distribute chunks across 4 GPUs
info = get_cvc_info("embeddings.cvc")
num_chunks = info['num_chunks']
chunks_per_gpu = num_chunks // 4

# GPU 0
chunks_gpu0 = list(range(0, chunks_per_gpu))
vectors_gpu0 = load_cvc_range("embeddings.cvc", chunks_gpu0, device="cuda:0")

# GPU 1
chunks_gpu1 = list(range(chunks_per_gpu, 2*chunks_per_gpu))
vectors_gpu1 = load_cvc_range("embeddings.cvc", chunks_gpu1, device="cuda:1")

# ... and so on
```

**Benefit:** 4x speedup via parallelization.

---

### Example 5: Inspecting Before Loading

**Before (blind loading):**

```python
from decompressed import load_cvc

# Don't know what we're loading until it's done
vectors = load_cvc("embeddings.cvc", device="cpu")
print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
```

**After (inspect first):**

```python
from decompressed import get_cvc_info, load_cvc

# Check metadata first
info = get_cvc_info("embeddings.cvc")
print(f"File contains {info['num_vectors']} vectors of dimension {info['dimension']}")
print(f"Compression: {info['compression']}, Chunks: {info['num_chunks']}")

# Decide whether to load based on metadata
if info['num_vectors'] < 1_000_000:
    vectors = load_cvc("embeddings.cvc", device="cpu")
else:
    print("File too large, using chunked loading...")
    # Use chunked API instead
```

**Benefit:** Make informed decisions before loading data.

## API Quick Reference

| Old API | New Chunked API | Use Case |
|---------|-----------------|----------|
| `load_cvc(path)` | `load_cvc_chunked(path)` | Memory-efficient iteration |
| - | `load_cvc_range(path, chunks)` | Load specific chunks |
| - | `get_cvc_info(path)` | Inspect before loading |

## Best Practices

### 1. Check File Info First

```python
info = get_cvc_info("embeddings.cvc")
if info['num_vectors'] > MEMORY_THRESHOLD:
    # Use chunked loading
    for chunk_idx, vectors in load_cvc_chunked(...):
        process(vectors)
else:
    # Use standard loading
    vectors = load_cvc(...)
    process(vectors)
```

### 2. Free Memory Between Chunks

```python
import gc

for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc"):
    process(vectors)
    del vectors  # Explicitly free memory
    gc.collect()  # Force garbage collection
```

### 3. Use Chunk Indices for Debugging

```python
# Process just one chunk for testing
for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc", chunk_indices=[0]):
    test_processing_pipeline(vectors)
```

### 4. Batch Processing

```python
# Process chunks in batches
batch_size = 3
info = get_cvc_info("embeddings.cvc")

for start in range(0, info['num_chunks'], batch_size):
    end = min(start + batch_size, info['num_chunks'])
    chunk_indices = list(range(start, end))
    batch = load_cvc_range("embeddings.cvc", chunk_indices=chunk_indices)
    process_batch(batch)
```

## Performance Tips

1. **Chunk size matters**: When creating `.cvc` files with `pack_cvc()`, choose `chunk_size` based on your typical access pattern:
   - Small chunks (10k-50k): Better for selective loading
   - Large chunks (100k-500k): Better for sequential processing

2. **GPU vs CPU**: Chunked loading works with both, but GPU decompression is much faster:
   ```python
   # Faster on GPU
   for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc", device="cuda"):
       ...
   ```

3. **Backend selection**: Use `backend="auto"` to automatically select the fastest backend.

## Common Patterns

### Pattern: Sliding Window

```python
from collections import deque

window = deque(maxlen=2)  # Keep last 2 chunks in memory
for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc"):
    window.append(vectors)
    if len(window) == 2:
        process_with_context(window[0], window[1])
```

### Pattern: Accumulator

```python
total_sum = 0
total_count = 0

for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc"):
    total_sum += vectors.sum()
    total_count += len(vectors)

mean = total_sum / total_count
```

### Pattern: Filter and Collect

```python
results = []
for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc"):
    # Filter vectors based on some criterion
    filtered = vectors[filter_criterion(vectors)]
    results.append(filtered)

final_results = np.concatenate(results, axis=0)
```

## Questions or Issues?

If you encounter any issues migrating to chunked decompression, please:
1. Check the examples in `examples/chunked_decompression.py`
2. Review the API documentation in `README.md`
3. Open an issue on GitHub with your use case
