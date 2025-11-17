# Decompressed Examples

This directory contains example scripts demonstrating how to use the Decompressed library.

## Available Examples

### `chunked_decompression.py`

Comprehensive demonstration of the chunked decompression APIs.

**Features demonstrated:**
- Inspecting file metadata with `get_cvc_info()`
- Iterating through chunks with `load_cvc_chunked()`
- Loading specific chunks with chunk indices
- Concatenating chunk ranges with `load_cvc_range()`
- Memory-efficient processing of large files
- Comparison with full file loading

**How to run:**

```bash
python examples/chunked_decompression.py
```

This will create a sample `.cvc` file and demonstrate all chunked decompression features.

## Quick Start

```python
import numpy as np
from decompressed import pack_cvc, load_cvc_chunked, get_cvc_info

# Create a sample file
embeddings = np.random.randn(100000, 768).astype(np.float32)
pack_cvc(embeddings, "sample.cvc", compression="fp16", chunk_size=20000)

# Inspect the file
info = get_cvc_info("sample.cvc")
print(f"File has {info['num_chunks']} chunks")

# Process chunks one at a time
for chunk_idx, vectors in load_cvc_chunked("sample.cvc"):
    print(f"Processing chunk {chunk_idx}: {vectors.shape}")
```

## More Examples Coming Soon

Additional examples will be added for:
- GPU decompression with different backends
- Integration with similarity search
- Distributed processing across multiple GPUs
- Streaming from remote sources
