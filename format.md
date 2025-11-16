# CVC File Format Specification

## Overview

The `.cvc` (Compressed Vector Collection) format is a binary file format designed for efficient storage and GPU-native decompression of vector embeddings. It supports multiple compression schemes optimized for different precision-performance tradeoffs.

## Design Goals

- **GPU-Native**: Direct decompression to GPU memory without CPU intermediate buffers
- **Streaming**: Chunked format allows partial loading and streaming decompression
- **Flexible**: Supports multiple compression schemes (FP16, INT8 quantization)
- **Metadata-Rich**: JSON header stores compression parameters and chunk information
- **Framework-Agnostic**: Works with PyTorch, CuPy, NumPy, and raw CUDA

## File Structure

```
┌─────────────────────────────────────────────┐
│ Magic Number (4 bytes)                      │  "CVCF"
├─────────────────────────────────────────────┤
│ Header Length (4 bytes, little-endian)      │
├─────────────────────────────────────────────┤
│ JSON Header (variable length)               │
│  - num_vectors                              │
│  - dimension                                │
│  - compression                              │
│  - chunks (array of chunk metadata)         │
├─────────────────────────────────────────────┤
│ Chunk 1 Length (4 bytes, little-endian)     │
├─────────────────────────────────────────────┤
│ Chunk 1 Payload (variable length)           │
│  - Compressed vector data                   │
├─────────────────────────────────────────────┤
│ Chunk 2 Length (4 bytes, little-endian)     │
├─────────────────────────────────────────────┤
│ Chunk 2 Payload (variable length)           │
├─────────────────────────────────────────────┤
│ ...                                         │
├─────────────────────────────────────────────┤
│ Chunk N Length (4 bytes, little-endian)     │
├─────────────────────────────────────────────┤
│ Chunk N Payload (variable length)           │
└─────────────────────────────────────────────┘
```

## Header Format

### Magic Number
- **Size**: 4 bytes
- **Value**: ASCII `"CVCF"` (`0x43 0x56 0x43 0x46`)
- **Purpose**: File type identification and validation

### Header Length
- **Size**: 4 bytes
- **Encoding**: Unsigned 32-bit integer, little-endian
- **Purpose**: Specifies the length of the JSON header in bytes

### JSON Header

A UTF-8 encoded JSON object containing file metadata:

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

#### Required Fields

- **`num_vectors`** (integer): Total number of vectors in the file
- **`dimension`** (integer): Dimensionality of each vector
- **`compression`** (string): Default compression scheme
  - Valid values: `"fp16"`, `"int8"`
- **`chunks`** (array): Array of chunk metadata objects

#### Chunk Metadata Object

Each chunk metadata object contains:

- **`rows`** (integer): Number of vectors in this chunk
- **`compression`** (string, optional): Compression scheme for this chunk (defaults to file-level compression)
- **`min`** (float, required for INT8): Minimum value for INT8 dequantization
- **`scale`** (float, required for INT8): Scale factor for INT8 dequantization

## Chunk Structure

Each chunk consists of:

1. **Chunk Length** (4 bytes, little-endian): Byte length of the compressed payload
2. **Compressed Payload** (variable): Binary compressed vector data

### Chunk Payload Encoding

#### FP16 Compression

- **Storage**: IEEE 754 half-precision (16-bit) floating-point
- **Layout**: Row-major, vectors stored sequentially
- **Size**: `rows × dimension × 2 bytes`
- **Decompression**: Direct hardware conversion (FP16 → FP32)

Example for 2 vectors of dimension 4:
```
[v0_d0, v0_d1, v0_d2, v0_d3, v1_d0, v1_d1, v1_d2, v1_d3]
```
Each element is 2 bytes (uint16 in FP16 format).

#### INT8 Quantization

- **Storage**: Unsigned 8-bit integers (0-255)
- **Layout**: Row-major, vectors stored sequentially
- **Size**: `rows × dimension × 1 byte`
- **Dequantization Formula**: 
  ```
  float_value = (uint8_value × scale) + min
  ```
- **Parameters**:
  - `min`: Minimum value of the original float range
  - `scale`: Quantization scale, typically `(max - min) / 255`

Example for 2 vectors of dimension 4:
```
[v0_d0, v0_d1, v0_d2, v0_d3, v1_d0, v1_d1, v1_d2, v1_d3]
```
Each element is 1 byte (uint8).

## Compression Schemes

### FP16 (Half-Precision)

**Characteristics:**
- 2× size reduction vs FP32
- Hardware-accelerated on modern GPUs (Tensor Cores, FP16 ALUs)
- Minimal accuracy loss for most ML embeddings
- Dynamic range: ±65504, precision: ~3 decimal digits

**Use Case:** General-purpose compression for embeddings where quality is critical

**Compression Ratio:** 2:1

### INT8 Quantization

**Characteristics:**
- 4× size reduction vs FP32
- Requires calibration (min/max computation)
- Per-chunk quantization parameters
- Linear quantization with affine transformation

**Use Case:** Maximum compression for large-scale similarity search where small precision loss is acceptable

**Compression Ratio:** 4:1

**Quantization Process:**
1. Compute `min` and `max` over the chunk
2. Calculate `scale = (max - min) / 255`
3. Quantize: `uint8_value = round((float_value - min) / scale)`

## Chunking Strategy

Chunks allow:
- **Streaming decompression**: Load and decompress data in batches
- **Mixed compression**: Different chunks can use different schemes
- **Memory efficiency**: Process large datasets without loading everything into memory

**Recommended Chunk Sizes:**
- Small: 10,000 - 50,000 vectors (good for streaming)
- Medium: 100,000 - 500,000 vectors (balanced)
- Large: 1,000,000+ vectors (minimize overhead)

## Usage Examples

### Creating a CVC File

```python
import numpy as np
from decompressed import pack_cvc

# Generate sample embeddings
embeddings = np.random.randn(1000000, 768).astype(np.float32)

# Pack with FP16 compression
pack_cvc(
    embeddings,
    output_path="embeddings.cvc",
    compression="fp16",
    chunk_size=100000
)

# Pack with INT8 compression
pack_cvc(
    embeddings,
    output_path="embeddings_int8.cvc",
    compression="int8",
    chunk_size=100000
)
```

### Loading a CVC File

```python
from decompressed import load_cvc

# Load to CPU (NumPy)
vectors_cpu = load_cvc("embeddings.cvc", device="cpu")

# Load to GPU (CuPy)
vectors_gpu = load_cvc("embeddings.cvc", device="cuda", framework="cupy")

# Load to GPU (PyTorch)
vectors_torch = load_cvc("embeddings.cvc", device="cuda", framework="torch")
```

## Performance Characteristics

### Compression Ratios

| Compression | Size vs FP32 | Typical Use Case |
|-------------|--------------|------------------|
| FP16        | 50%          | High-quality embeddings |
| INT8        | 25%          | Large-scale similarity search |

### Decompression Throughput (A100 GPU)

| Compression | Throughput    | Notes |
|-------------|---------------|-------|
| FP16        | ~500 GB/s     | Hardware FP16 conversion |
| INT8        | ~800 GB/s     | Simple INT8→FP32 kernel |

### Memory Bandwidth Comparison

For 1M vectors × 768 dimensions (3GB uncompressed):

| Format      | Storage Size | Load Time (PCIe Gen4) |
|-------------|--------------|------------------------|
| FP32        | 3.0 GB       | ~1.2 seconds          |
| FP16 (CVC)  | 1.5 GB       | ~0.6 seconds          |
| INT8 (CVC)  | 0.75 GB      | ~0.3 seconds          |

## Implementation Notes

### Endianness
- All multi-byte integers use **little-endian** byte order
- FP16 values follow IEEE 754 half-precision format (little-endian)

### Alignment
- No specific alignment requirements
- Sequential byte-packing for maximum space efficiency

### Error Handling
- Invalid magic number should raise a clear error
- Mismatched chunk count vs. header should be validated
- Truncated files should be detected during chunk reading

### Extensions

The format is designed to be extensible:
- New compression schemes can be added (e.g., `"int4"`, `"bfloat16"`)
- Additional metadata fields in JSON header are ignored by older readers
- Chunk-level metadata can be extended without breaking compatibility

## References

- IEEE 754 Half-Precision: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
- Quantization Techniques: https://arxiv.org/abs/2106.08295

## Version History

- **v0.1.0**: Initial format specification
  - FP16 and INT8 compression
  - JSON metadata header
  - Chunked storage

---

**License**: Apache 2.0  
**Specification Version**: 0.1.0  
**Last Updated**: 2025-11-15
