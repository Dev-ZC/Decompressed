# CVC File Format Specification

**Version 2.0** - Multi-Column Storage, Zero-Copy, and GPU Optimization

The `.cvc` (Compressed Vector Collection) format is a binary container for large collections of dense embeddings.  
It is designed to support:

- **Multi-column storage**: Heterogeneous tensors in one file (v2.0)
- **Efficient compression**: FP16 (2×) and INT8 (4×) compression
- **GPU-native decompression**: 20-80× faster than CPU
- **Zero-copy memory mapping**: 50-90% memory reduction
- **Selective column loading**: Load 1 of N columns
- **Section metadata**: Multi-source data with filtering
- **Streaming/chunked**: Independent chunk decompression

This document specifies both v1.x (single-column) and v2.0 (multi-column) formats.

---

## Design Goals

- **Multi-modal ready** (v2.0)  
  Store text, image, audio embeddings in one file with heterogeneous dimensions

- **GPU-native**  
  Triton and CUDA kernels decompress directly into GPU memory

- **Zero-copy access**  
  Memory-mapped files with page alignment for minimal memory footprint

- **Selective loading**  
  Load 1 of 10 columns without reading entire file (10× I/O reduction)

- **Streaming / chunked**  
  Independently decompressible chunks for partial loading

- **Multiple compression schemes**  
  Per-column FP16, INT8, or no compression

- **Metadata-rich**  
  JSON header with schema, sections, and extensible metadata

- **Framework-agnostic**  
  Works with NumPy, PyTorch, CuPy

---

## File structure

A `.cvc` file has the following high-level structure:

```text
┌─────────────────────────────────────────────┐
│ Magic Number (4 bytes)                     │  "CVCF"
├─────────────────────────────────────────────┤
│ Header Length (4 bytes, little-endian)     │  uint32
├─────────────────────────────────────────────┤
│ JSON Header (variable length)              │
│  - num_vectors                             │
│  - dimension                               │
│  - compression                             │
│  - chunks (array of per-chunk metadata)    │
├─────────────────────────────────────────────┤
│ Chunk 1 Length (4 bytes, little-endian)    │  uint32
├─────────────────────────────────────────────┤
│ Chunk 1 Payload (variable length)          │
├─────────────────────────────────────────────┤
│ Chunk 2 Length (4 bytes, little-endian)    │
├─────────────────────────────────────────────┤
│ Chunk 2 Payload (variable length)          │
├─────────────────────────────────────────────┤
│ ...                                       │
├─────────────────────────────────────────────┤
│ Chunk N Length (4 bytes, little-endian)    │
├─────────────────────────────────────────────┤
│ Chunk N Payload (variable length)          │
└─────────────────────────────────────────────┘
```

---

## Header format

### Magic number

- **Size**: 4 bytes  
- **Value**: ASCII `"CVCF"` (`0x43 0x56 0x43 0x46`)  
- **Purpose**: File identification and basic validation.

### Header length

- **Size**: 4 bytes  
- **Encoding**: Unsigned 32-bit integer, little-endian  
- **Purpose**: Number of bytes in the JSON header that follows.

### JSON header

The JSON header is UTF‑8 encoded and contains at least:

```json
{
  "num_vectors": 1000000,
  "dimension": 768,
  "compression": "fp16",
  "chunks": [
    {
      "rows": 100000,
      "compression": "fp16"
    },
    {
      "rows": 100000,
      "compression": "int8",
      "min": -0.5,
      "scale": 0.00392156862
    }
  ]
}
```

#### Required fields

- `num_vectors` (`int`): total number of vectors in the file.
- `dimension` (`int`): dimensionality of each vector.
- `compression` (`str`): default compression scheme for chunks.
  - Currently supported: `"fp16"`, `"int8"`.
- `chunks` (`list`): array of per‑chunk metadata objects. Each element describes one chunk appearing later in the file.

#### Chunk metadata

Each chunk entry is a JSON object with:

- `rows` (`int`): number of vectors stored in this chunk.
- `compression` (`str`, optional): compression scheme for the chunk. If omitted, defaults to the file-level `compression`.
- `min` (`float`, required for `int8`): minimum original value used for quantization in this chunk.
- `scale` (`float`, required for `int8`): scale factor used for dequantization.

The `min` and `scale` fields are used by the INT8 decompression kernels to reconstruct approximate FP32 values.

---

## Chunk structure

Each chunk is encoded as:

1. **Chunk length** (4 bytes, little-endian): length in bytes of the compressed payload.
2. **Chunk payload** (variable length): compressed vector data.

### FP16 payload

- **Storage**: IEEE 754 half-precision (16‑bit) floats.
- **Layout**: row-major, contiguous:
  ```text
  [v0_d0, v0_d1, ..., v0_d{dim-1}, v1_d0, ..., v1_d{dim-1}, ...]
  ```
- **Size**: `rows * dimension * 2` bytes.
- **Decompression**:
  - CPU: FP16 to FP32 conversion in Python/C++.
  - GPU (Triton/CUDA): FP16 to FP32 conversion in dedicated kernels.

### INT8 payload

- **Storage**: unsigned 8-bit integers (`uint8`).
- **Layout**: row-major, same layout as FP16, but 1 byte per element.
- **Size**: `rows * dimension * 1` byte.
- **Dequantization**:
  ```text
  float_value = (uint8_value * scale) + min
  ```
  where `min` and `scale` are taken from the chunk metadata.

The INT8 scheme is linear and uses per-chunk parameters to adapt to local value ranges.

---

## Compression schemes

### FP16 (half-precision)

- **Characteristics**

  - 2× size reduction vs FP32.
  - Hardware-accelerated on modern GPUs (e.g. Tensor Cores).
  - Good default option for high-quality embeddings.

- **Compression ratio**

  - Approximately **2:1** compared to FP32.

### INT8 (linear quantization)

- **Characteristics**

  - 4× size reduction vs FP32.
  - Requires computing `min`/`max` (and thus `scale`) per chunk.
  - Single affine transformation to reconstruct approximate FP32 values.

- **Compression ratio**

  - Approximately **4:1** compared to FP32.

- **Quantization process**

  1. Compute `min` and `max` for the chunk (over all values).
  2. Compute `scale = (max - min) / 255`.
  3. For each value `x`, compute:
     ```text
     q = round((x - min) / scale)
     ```
     and clamp to `[0, 255]`.
  4. Store `q` as `uint8`, along with `min` and `scale` in the chunk metadata.

---

## Chunking strategy

Chunking is a key mechanism for scalability and streaming:

- **Streaming decompression**

  - Chunks can be read and decompressed incrementally.
  - You can load only a subset of the dataset if needed.

- **Mixed compression**

  - Different chunks may use different compression schemes.
  - For example, earlier chunks may use INT8 for less critical data, while later chunks use FP16.

- **Memory efficiency**

  - On both CPU and GPU, chunks can be processed sequentially to bound peak memory usage.

### Recommended chunk sizes

Typical ranges (not enforced by the format):

- Small: 10k – 50k vectors (good for low-latency streaming).
- Medium: 100k – 500k vectors (balanced).
- Large: ≥ 1M vectors (minimizes header overhead, good for offline processing).

These are exposed via the `chunk_size` argument of `pack_cvc`.

---

## Integration with the Decompressed API

### Creating `.cvc` files

Use `pack_cvc` to construct compliant `.cvc` files:

```python
import numpy as np
from decompressed import pack_cvc

embeddings = np.random.randn(1_000_000, 768).astype(np.float32)

# FP16
pack_cvc(
    embeddings,
    output_path="embeddings_fp16.cvc",
    compression="fp16",
    chunk_size=100_000,
)

# INT8
pack_cvc(
    embeddings,
    output_path="embeddings_int8.cvc",
    compression="int8",
    chunk_size=100_000,
)
```

This function:

- Computes per-chunk metadata (including INT8 `min`/`scale` when needed).
- Builds the JSON header.
- Writes the magic number, header, and each chunk’s length and payload.

### Loading `.cvc` files

Use `load_cvc` to read and decompress:

```python
from decompressed import load_cvc

# CPU (NumPy)
vectors_cpu = load_cvc("embeddings_fp16.cvc", device="cpu")

# GPU (PyTorch + Triton backend)
vectors_torch = load_cvc(
    "embeddings_fp16.cvc",
    device="cuda",
    framework="torch",
    backend="auto",  # prefers CUDA, falls back to Triton
)

# GPU (CuPy)
vectors_cupy = load_cvc(
    "embeddings_fp16.cvc",
    device="cuda",
    framework="cupy",
    backend="triton",
)
```

The loader:

1. Validates the magic number and parses the JSON header.
2. Allocates an output array of shape `(num_vectors, dimension)` on CPU or GPU.
3. Chooses a backend based on `device`, `backend`, and availability.
4. Iterates over chunks, reading each payload and invoking the backend’s `decompress_chunk` implementation.

---

## Performance characteristics

Approximate characteristics (hardware-dependent):

### Compression ratios

| Compression | Size vs FP32 | Typical use case                      |
|------------|--------------|----------------------------------------|
| FP16       | ~50%         | High-quality embedding storage         |
| INT8       | ~25%         | Large-scale similarity search, recall-tolerant |

### Decompression throughput (indicative)

On modern data center GPUs:

| Compression | Backend  | Throughput (approximate)  |
|------------|----------|---------------------------|
| FP16       | Triton   | O(100s) GB/s              |
| INT8       | Triton   | O(100s) GB/s              |
| FP16/INT8  | CUDA (*) | Target: ≥ Triton on NVIDIA|

\* CUDA native backend is under development; concrete numbers will depend on the final implementation and hardware.

---

## Implementation notes

### Endianness

- All multi-byte integers (header length, chunk lengths) use **little-endian**.
- FP16 and FP32 values are stored in the native little-endian layout.

### Alignment

- The format does not enforce additional alignment constraints.
- Payloads are packed sequentially for maximal space efficiency.

### Error handling

Readers should:

- Reject files with an invalid magic number.
- Validate that the number of chunks read matches the header.
- Detect truncated files (e.g., chunk length exceeding remaining bytes).

The Python loader raises descriptive exceptions in these cases.

---

## Extensibility

The format is designed to evolve:

- New compression schemes (e.g. `"int4"`, `"bfloat16"`) can be introduced by:
  - Adding new `compression` values.
  - Adding appropriate per-chunk metadata fields.
- Extra fields in the JSON header should be ignored by older readers.
- Chunk metadata objects can be extended with additional keys without breaking compatibility.

Backends should treat unknown `compression` values as unsupported and raise a clear error.

---

## References

- IEEE 754 Half-Precision: <https://en.wikipedia.org/wiki/Half-precision_floating-point_format>  
- Quantization Techniques: <https://arxiv.org/abs/2106.08295>

---

## Format Versions

### v1.x (Single-Column Format)

Original format with single vector dimension:
- One dimension per file
- Global compression setting
- Section metadata support

### v2.0 (Multi-Column Format) 🆕

Extended format for heterogeneous data:
```json
{
  "format_type": "columnar",
  "num_vectors": 1000000,
  "columns": [
    {
      "name": "text",
      "dimension": 768,
      "dtype": "float32",
      "compression": "fp16"
    },
    {
      "name": "image",
      "dimension": 512,
      "dtype": "float32",
      "compression": "int8"
    },
    {
      "name": "doc_id",
      "dimension": 1,
      "dtype": "int32",
      "compression": "none"
    }
  ],
  "chunks": [
    {
      "rows": 100000,
      "columns": {
        "text": {"offset": 0, "size": 153600000, "compression": "fp16"},
        "image": {"offset": 153600000, "size": 51200000, "compression": "int8", "min": -1.0, "scale": 0.00784},
        "doc_id": {"offset": 204800000, "size": 400000, "compression": "none"}
      }
    }
  ]
}
```

**Key differences from v1.x:**
- `format_type: "columnar"` indicates v2.0
- `columns` array defines schema for each column
- Per-column compression settings
- Chunk payload contains multiple columns at specified offsets
- Supports heterogeneous dimensions and data types

### v2.0 with MMap Optimization

Files packed with `mmap_optimized=True` add:
```json
{
  "mmap_optimized": true,
  "chunks": [
    {
      "file_offset": 4096,  // Pre-computed file position
      "rows": 100000,
      // ...
    }
  ]
}
```

**Benefits:**
- 4KB page-aligned chunks
- Direct memory-mapped access
- ~5% file size increase
- 50-90% memory reduction during loading

---

## Version History

- **v2.0** (November 2024)
  - Multi-column storage with heterogeneous tensors
  - Per-column compression settings
  - Schema introspection
  - Memory-mapped file support with page alignment
  - Section metadata for multi-source data
  - GPU-accelerated columnar loading

- **v1.0** (2024)
  - Format versioning (major.minor)
  - Backward compatibility with v0.x
  - Section metadata support
  - GPU backend integration

- **v0.1.0** (Initial)
  - Initial specification with FP16 and INT8 compression
  - JSON metadata header
  - Chunked storage format

---

**License**: Apache 2.0  
**Specification version**: 2.0  
**Last updated**: 2024-11-27
