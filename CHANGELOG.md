# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - v2.0.0

### Added - Multi-Column Storage (Major Feature)

**🎉 CVC v2.0: Columnar storage for heterogeneous tensors**

- **`pack_cvc_columns()`** - Pack multiple arrays with different dimensions in one file
  - Store text + image + metadata together
  - Per-column compression control (fp16, int8, none)
  - Automatic schema validation
  
- **`load_cvc_columns()`** - Selective column loading
  - Load only requested columns (2-5× faster)
  - Section filtering support
  - Combined section + column filtering
  
- **`pack_cvc_sections_columnar()`** - Combine sections + columns
  - Multi-modal data from multiple sources
  - Vertical stacking (sections) + horizontal expansion (columns)
  
### New Modules

- `python/decompressed/schema.py` (200 lines) - Column schema definitions and validation
- `python/decompressed/columnar.py` (300 lines) - Multi-column packer
- `python/decompressed/columnar_loader.py` (250 lines) - Selective column loader
- `tests/test_columnar.py` (250 lines) - Comprehensive tests (10 tests, all passing)
- `examples/columnar_example.py` (100 lines) - Usage examples

### Performance

- **Selective loading**: 2-5× faster when loading subset of columns
- **Storage overhead**: <1% for column metadata
- **Compression**: Same ratios as v1.0 (per-column control)

### Backward Compatibility

- ✅ 100% backward compatible with v1.x files
- ✅ v1.x API (`load_cvc`, `pack_cvc`) unchanged
- ✅ v2.x API (`load_cvc_columns`, `pack_cvc_columns`) is additive
- ✅ Auto-detection of format version in loader

### Documentation

- Updated `ADVANCED_FEATURES_DESIGN.md` with sections integration
- Created `IMPLEMENTATION_STATUS.md` with complete implementation details
- Updated `README.md` with multi-column examples

---

## [0.1.0] - 2024-XX-XX

### Initial Release

- FP16 and INT8 compression
- CPU (Python, C++) and GPU (CUDA, Triton) backends
- Chunked streaming support
- Section-based metadata filtering
- Format versioning (v1.0)
