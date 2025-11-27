"""Multi-column CVC loader with selective column loading."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zlib

from .decompress import decompress_fp16_cpu, decompress_int8_cpu
from .schema import ColumnarFileSchema
from .backends.cpu import PythonBackend, CPPBackend
from .backends.gpu import CUDABackend, TritonBackend
from .utils import select_backend, validate_backend_availability

HEADER_MAGIC = b"CVCF"


class ColumnarCVCLoader:
    """Loader for multi-column CVC files with selective column loading."""
    
    def __init__(self):
        """Initialize loader with all available backends."""
        self.python_backend = PythonBackend()
        self.cpp_backend = CPPBackend()
        self.cuda_backend = CUDABackend()
        self.triton_backend = TritonBackend()
    
    def get_backend_availability(self):
        """Check which backends are available."""
        return {
            'python': True,  # Always available
            'cpp': self.cpp_backend.is_available(),
            'cuda': self.cuda_backend.is_available(),
            'triton': self.triton_backend.is_available(),
        }
    
    def load(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        device: str = "cpu",
        framework: str = "torch",
        backend: str = "auto",
        section_key: Optional[str] = None,
        section_value: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load columnar CVC file with selective column and section loading.
        
        Args:
            path: Path to columnar .cvc file
            columns: List of column names to load, or None for all
            device: "cpu" or "cuda"
            framework: "torch" or "cupy"
            backend: Backend to use - "auto", "python", "cpp", "cuda", or "triton"
            section_key: Optional section metadata key to filter by
            section_value: Value to match for section_key
        
        Returns:
            Dict mapping column names to arrays
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            header = self._read_header(f)
            
            if header.get("format_type") != "columnar":
                raise ValueError(
                    f"File is not columnar format. Use load_cvc() for single-column files."
                )
            
            schema = ColumnarFileSchema.from_dict(header)
            
            # Determine what to load
            all_columns = [col.name for col in schema.columns]
            columns_to_load = columns if columns else all_columns
            
            # Validate requested columns
            for col in columns_to_load:
                if col not in all_columns:
                    raise ValueError(f"Column '{col}' not found. Available: {all_columns}")
            
            # Handle section filtering
            chunks_to_load, section_ranges = self._get_chunk_ranges(
                header, section_key, section_value
            )
            
            # Calculate output size
            num_vectors = self._calculate_output_size(
                header, chunks_to_load, section_ranges
            )
            
            # Select and validate backend
            availability = self.get_backend_availability()
            use_backend = select_backend(
                backend, device,
                availability['cpp'],
                availability['cuda'],
                availability['triton']
            )
            
            validate_backend_availability(
                use_backend, device,
                availability['cpp'],
                availability['cuda'],
                availability['triton']
            )
            
            # Allocate output
            column_map = {col.name: col for col in schema.columns}
            output = self._allocate_arrays(
                columns_to_load, column_map, num_vectors, device, framework
            )
            
            # Load data
            self._decompress_chunks(
                f, header, columns_to_load, column_map,
                chunks_to_load, section_ranges, output,
                device, framework, use_backend
            )
        
        return output
    
    def _read_header(self, f) -> dict:
        """Read and validate CVC v2.0 header."""
        magic = f.read(4)
        if magic != HEADER_MAGIC:
            raise ValueError("Not a valid .cvc file")
        
        major = int.from_bytes(f.read(2), "little")
        minor = int.from_bytes(f.read(2), "little")
        
        if major < 2:
            raise ValueError(
                f"File is v{major}.{minor} format. "
                f"Columnar format requires v2.0+"
            )
        
        header_len = int.from_bytes(f.read(4), "little")
        return json.loads(f.read(header_len))
    
    def _get_chunk_ranges(
        self,
        header: dict,
        section_key: Optional[str],
        section_value: Optional[str]
    ) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """Determine which chunks to load and row ranges within them."""
        if section_key is None:
            # Load all chunks fully
            return list(range(len(header["chunks"]))), []
        
        # Section filtering
        if "sections" not in header:
            raise ValueError(
                "File does not have section metadata. "
                "Use pack_cvc_sections_columnar() to create files with sections."
            )
        
        chunks_to_load = []
        section_ranges = []
        
        for chunk_idx, chunk in enumerate(header["chunks"]):
            if "sections" in chunk:
                for chunk_section in chunk["sections"]:
                    if chunk_section.get("metadata", {}).get(section_key) == section_value:
                        chunks_to_load.append(chunk_idx)
                        section_ranges.append((
                            chunk_idx,
                            chunk_section["start_in_chunk"],
                            chunk_section["end_in_chunk"]
                        ))
                        break
        
        if not chunks_to_load:
            raise ValueError(f"No chunks found with {section_key}={section_value}")
        
        return chunks_to_load, section_ranges
    
    def _calculate_output_size(
        self,
        header: dict,
        chunks_to_load: List[int],
        section_ranges: List[Tuple[int, int, int]]
    ) -> int:
        """Calculate total output vector count."""
        if section_ranges:
            # Sum up section ranges
            return sum(end - start for _, start, end in section_ranges)
        else:
            # Sum up full chunks
            return sum(header["chunks"][i]["rows"] for i in chunks_to_load)
    
    def _allocate_arrays(
        self,
        columns: List[str],
        column_map: Dict,
        num_vectors: int,
        device: str,
        framework: str
    ) -> Dict[str, np.ndarray]:
        """Allocate output arrays for each column."""
        output = {}
        
        for col_name in columns:
            col = column_map[col_name]
            dim = col.dimension
            
            if device == "cpu":
                shape = (num_vectors,) if dim == 1 else (num_vectors, dim)
                output[col_name] = np.empty(shape, dtype=np.float32)
            else:
                if framework == "torch":
                    import torch
                    shape = (num_vectors,) if dim == 1 else (num_vectors, dim)
                    output[col_name] = torch.zeros(shape, dtype=torch.float32, device="cuda")
                elif framework == "cupy":
                    import cupy as cp
                    shape = (num_vectors,) if dim == 1 else (num_vectors, dim)
                    output[col_name] = cp.zeros(shape, dtype=cp.float32)
        
        return output
    
    def _decompress_chunks(
        self,
        f,
        header: dict,
        columns_to_load: List[str],
        column_map: Dict,
        chunks_to_load: List[int],
        section_ranges: List[Tuple[int, int, int]],
        output: Dict,
        device: str,
        framework: str,
        backend: str
    ):
        """Read and decompress requested chunks and columns."""
        chunks_meta = header["chunks"]
        output_offset = 0
        
        for chunk_idx in range(len(chunks_meta)):
            chunk = chunks_meta[chunk_idx]
            chunk_len = int.from_bytes(f.read(4), "little")
            expected_checksum = int.from_bytes(f.read(4), "little")
            
            if chunk_idx in chunks_to_load:
                # Read and validate
                payload = f.read(chunk_len)
                actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
                
                if actual_checksum != expected_checksum:
                    raise ValueError(
                        f"Chunk {chunk_idx} corrupted: checksum mismatch"
                    )
                
                rows = chunk["rows"]
                
                # Determine row range to extract
                if section_ranges:
                    # Find section range for this chunk
                    row_start, row_end = None, None
                    for range_chunk_idx, start, end in section_ranges:
                        if range_chunk_idx == chunk_idx:
                            row_start, row_end = start, end
                            break
                    
                    if row_start is None:
                        continue  # Skip this chunk
                else:
                    row_start, row_end = 0, rows
                
                # Decompress each requested column
                for col_name in columns_to_load:
                    col = column_map[col_name]
                    col_chunk_meta = chunk["columns"][col_name]
                    
                    # Extract column data from payload
                    col_offset = col_chunk_meta["offset"]
                    col_size = col_chunk_meta["size"]
                    col_payload = payload[col_offset:col_offset + col_size]
                    
                    # Decompress (GPU or CPU)
                    chunk_data = self._decompress_column(
                        col_payload, col_chunk_meta, rows, col.dimension,
                        device, framework, backend
                    )
                    
                    # Extract section range if needed
                    if row_start > 0 or row_end < rows:
                        chunk_data = chunk_data[row_start:row_end]
                    
                    # Copy to output
                    rows_to_copy = row_end - row_start
                    if col.dimension == 1:
                        output[col_name][output_offset:output_offset+rows_to_copy] = chunk_data.flatten()
                    else:
                        output[col_name][output_offset:output_offset+rows_to_copy] = chunk_data
                
                output_offset += row_end - row_start
            else:
                # Skip this chunk
                f.seek(chunk_len, 1)
    
    def _decompress_column(
        self,
        payload: bytes,
        col_meta: dict,
        rows: int,
        dimension: int,
        device: str,
        framework: str,
        backend: str
    ) -> np.ndarray:
        """Decompress a single column's data with GPU or CPU backend."""
        compression = col_meta["compression"]
        
        # For GPU backends, use the backend's decompress method
        if backend in ["cuda", "triton"] and device == "cuda":
            # Allocate output on GPU
            if framework == "torch":
                import torch
                if dimension == 1:
                    output = torch.empty(rows, dtype=torch.float32, device="cuda")
                else:
                    output = torch.empty((rows, dimension), dtype=torch.float32, device="cuda")
            elif framework == "cupy":
                import cupy as cp
                if dimension == 1:
                    output = cp.empty(rows, dtype=cp.float32)
                else:
                    output = cp.empty((rows, dimension), dtype=cp.float32)
            
            # Get backend instance
            backend_instance = self._get_backend_instance(backend)
            
            # Decompress on GPU
            if compression == "fp16":
                backend_instance.decompress_chunk(
                    payload, rows, dimension, "fp16", {},
                    output, 0, framework=framework
                )
            elif compression == "int8":
                minv = col_meta["min"]
                scale = col_meta["scale"]
                chunk_meta = {"min": minv, "scale": scale}
                backend_instance.decompress_chunk(
                    payload, rows, dimension, "int8", chunk_meta,
                    output, 0, framework=framework
                )
            elif compression == "none":
                # Uncompressed - copy to GPU
                data = np.frombuffer(payload, dtype=np.int32)
                if dimension > 1:
                    data = data.reshape(rows, dimension)
                data = data.astype(np.float32)
                if framework == "torch":
                    import torch
                    output = torch.from_numpy(data).cuda()
                elif framework == "cupy":
                    import cupy as cp
                    output = cp.asarray(data)
            
            return output
        
        # CPU path (existing code)
        if compression == "fp16":
            return decompress_fp16_cpu(payload, rows, dimension)
        
        elif compression == "int8":
            minv = col_meta["min"]
            scale = col_meta["scale"]
            return decompress_int8_cpu(payload, rows, dimension, minv, scale)
        
        elif compression == "none":
            # Raw bytes
            data = np.frombuffer(payload, dtype=np.int32)
            if dimension > 1:
                data = data.reshape(rows, dimension)
            return data.astype(np.float32)
        
        else:
            raise ValueError(f"Unknown compression: {compression}")
    
    def _get_backend_instance(self, backend_name: str):
        """Get backend instance by name."""
        backend_map = {
            'python': self.python_backend,
            'cpp': self.cpp_backend,
            'cuda': self.cuda_backend,
            'triton': self.triton_backend,
        }
        return backend_map[backend_name]
