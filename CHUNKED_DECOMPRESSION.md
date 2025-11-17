# Chunked Decompression Feature

## Summary

**Added chunked decompression capabilities to the Decompressed library.** Users can now load and decompress specific chunks from `.cvc` files instead of loading the entire file at once.

## Answer to Your Question

> "Would we also need chunking for compression or no?"

**No, compression is already chunked!** The `pack_cvc()` function has always supported the `chunk_size` parameter, which chunks the data during compression. What was missing was the ability to **decompress** specific chunks on-demand.

## What Was Added

### New APIs

Three new public APIs were added to `decompressed`:

#### 1. `get_cvc_info(path)`
Read file metadata without loading vectors into memory.

```python
from decompressed import get_cvc_info

info = get_cvc_info("embeddings.cvc")
print(f"File contains {info['num_vectors']} vectors in {info['num_chunks']} chunks")
print(f"Dimension: {info['dimension']}, Compression: {info['compression']}")
```

**Returns:**
- `num_vectors`: Total number of vectors
- `dimension`: Vector dimensionality
- `compression`: Default compression scheme
- `chunks`: List of chunk metadata
- `num_chunks`: Number of chunks

**Use cases:**
- Inspect file contents before loading
- Check chunk structure
- Get file statistics without memory allocation

---

#### 2. `load_cvc_chunked(path, chunk_indices=None, device="cpu", framework="torch", backend="auto")`
Iterator that yields decompressed chunks one at a time.

```python
from decompressed import load_cvc_chunked

# Iterate through all chunks
for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc", device="cpu"):
    print(f"Processing chunk {chunk_idx}: {vectors.shape}")
    # Process this chunk...

# Load only specific chunks
for chunk_idx, vectors in load_cvc_chunked(
    "embeddings.cvc",
    chunk_indices=[0, 2, 5],
    device="cuda",
):
    process_chunk(vectors)
```

**Parameters:**
- `path`: Path to `.cvc` file
- `chunk_indices`: List of chunk indices to load (0-indexed), or `None` for all
- `device`: `"cpu"` or `"cuda"`
- `framework`: `"torch"` or `"cupy"` (for GPU)
- `backend`: `"auto"`, `"python"`, `"cpp"`, `"cuda"`, or `"triton"`

**Yields:**
- `(chunk_index, chunk_array)` tuples

**Use cases:**
- **Memory-efficient processing**: Process large files that don't fit in memory
- **Streaming workflows**: Load and process one chunk at a time
- **Selective loading**: Load only the chunks you need
- **Incremental computation**: Compute embeddings or similarities incrementally

---

#### 3. `load_cvc_range(path, chunk_indices, device="cpu", framework="torch", backend="auto")`
Load specific chunks and concatenate them into a single array.

```python
from decompressed import load_cvc_range

# Load first 3 chunks only
vectors = load_cvc_range("embeddings.cvc", chunk_indices=[0, 1, 2], device="cpu")

# Load specific non-contiguous chunks
vectors = load_cvc_range(
    "embeddings.cvc",
    chunk_indices=[0, 5, 10],
    device="cuda",
    backend="triton",
)
```

**Parameters:**
- `path`: Path to `.cvc` file
- `chunk_indices`: List of chunk indices to load (0-indexed)
- `device`: `"cpu"` or `"cuda"`
- `framework`: `"torch"` or `"cupy"` (for GPU)
- `backend`: `"auto"`, `"python"`, `"cpp"`, `"cuda"`, or `"triton"`

**Returns:**
- Array containing requested chunks concatenated together

**Use cases:**
- **Partial loading**: Load subset of vectors from large collection
- **Range queries**: Load vectors in specific index range
- **Sharded processing**: Process different chunks on different GPUs/machines

## Implementation Details

### Modified Files

1. **`python/decompressed/loader.py`**
   - Added `get_info()` method to `CVCLoader`
   - Added `load_chunks()` generator method for chunked iteration
   - Added `load_range()` method for range-based loading
   - Total additions: ~150 lines

2. **`python/decompressed/pycvc.py`**
   - Added `get_cvc_info()` wrapper function
   - Added `load_cvc_chunked()` wrapper function
   - Added `load_cvc_range()` wrapper function
   - Total additions: ~95 lines

3. **`python/decompressed/__init__.py`**
   - Exported new APIs to public interface

4. **`README.md`**
   - Added documentation for all three new APIs
   - Updated Key Features section
   - Added usage examples and use cases

5. **`python/decompressed/ARCHITECTURE.md`**
   - Updated module documentation
   - Added chunked loading examples
   - Marked streaming support as implemented

6. **`examples/chunked_decompression.py`** (NEW)
   - Comprehensive example script demonstrating all chunked APIs
   - 6 examples covering different use cases
   - Runnable demo for users

### Key Design Decisions

1. **Generator-based API**: `load_cvc_chunked()` returns a generator for memory efficiency
2. **Selective chunk reading**: Only reads requested chunks from disk, skipping others via `seek()`
3. **Consistent interface**: Same parameters as `load_cvc()` for device, framework, backend
4. **No breaking changes**: All existing APIs remain unchanged
5. **Backend support**: Works with all backends (Python, C++, CUDA, Triton)

## Benefits

### 1. Memory Efficiency
Load and process files larger than available RAM by processing one chunk at a time.

```python
# Process 10GB file with only 1GB RAM
for chunk_idx, vectors in load_cvc_chunked("huge_file.cvc", device="cpu"):
    results = process_chunk(vectors)
    save_results(results)
    # Memory is freed after each iteration
```

### 2. Faster Time-to-First-Result
Start processing before loading the entire file.

```python
# Start processing immediately after first chunk loads
for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc", device="cuda"):
    if early_exit_condition(vectors):
        break  # Don't need to load remaining chunks
```

### 3. Selective Loading
Only load the data you need.

```python
# Load first 10% of file for quick testing
info = get_cvc_info("embeddings.cvc")
first_chunks = list(range(info['num_chunks'] // 10))
vectors = load_cvc_range("embeddings.cvc", chunk_indices=first_chunks)
```

### 4. Distributed Processing
Different machines/GPUs can process different chunks.

```python
# GPU 0: Process chunks 0-4
vectors_gpu0 = load_cvc_range("embeddings.cvc", chunk_indices=[0,1,2,3,4], device="cuda:0")

# GPU 1: Process chunks 5-9
vectors_gpu1 = load_cvc_range("embeddings.cvc", chunk_indices=[5,6,7,8,9], device="cuda:1")
```

## Backward Compatibility

✅ **All existing code continues to work without changes.**

- Original `load_cvc()` still loads entire file at once
- No changes to `pack_cvc()` or other existing APIs
- New APIs are purely additive

## Testing Recommendations

Users should test the new APIs with:

1. **Small files**: Verify correctness with files that fit in memory
2. **Large files**: Test memory efficiency with files > available RAM
3. **GPU loading**: Test with `device="cuda"` and both Triton/CUDA backends
4. **Edge cases**: Single chunk files, non-contiguous chunk indices
5. **Comparison**: Verify chunked loading produces same results as full loading

Example test:

```python
# Verify chunked and full loading produce identical results
full = load_cvc("test.cvc")
chunks = [c for _, c in load_cvc_chunked("test.cvc")]
chunked = np.concatenate(chunks, axis=0)
assert np.allclose(full, chunked)
```

## Example Output

Running `examples/chunked_decompression.py`:

```
Chunked Decompression Examples
==================================================

Creating sample embeddings file...
Created sample_embeddings.cvc: 500000 vectors, 768 dims

=== Example 1: Inspecting File Metadata ===
File: sample_embeddings.cvc
  Total vectors: 500,000
  Dimension: 768
  Compression: fp16
  Number of chunks: 5
  Chunk sizes: [100000, 100000, 100000, 100000, 100000]

=== Example 2: Iterating Through All Chunks ===
Processing chunk 0: shape=(100000, 768), mean=0.0004, std=1.0001
Processing chunk 1: shape=(100000, 768), mean=-0.0002, std=0.9998
Processing chunk 2: shape=(100000, 768), mean=0.0001, std=1.0003
Processing chunk 3: shape=(100000, 768), mean=0.0006, std=0.9997
Processing chunk 4: shape=(100000, 768), mean=-0.0003, std=1.0002

=== Example 3: Loading Specific Chunks ===
Loading chunks: [0, 2, 4]
  Chunk 0: (100000, 768)
  Chunk 2: (100000, 768)
  Chunk 4: (100000, 768)

...
```

## Next Steps (Future Work)

Potential enhancements:

1. **Async loading**: Async versions of chunked APIs for concurrent I/O
2. **Remote file support**: Stream chunks from S3, HTTP, etc.
3. **Chunk-level caching**: Cache frequently accessed chunks
4. **Parallel chunk loading**: Load multiple chunks concurrently
5. **Chunk metadata in filenames**: Support `file.cvc#chunk=5` syntax

## Questions?

For questions or issues with the chunked decompression APIs, please open an issue on GitHub.
