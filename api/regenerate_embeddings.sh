#!/bin/bash
# Regenerate embeddings with correct dimensions
# This fixes dimension mismatch errors between query and stored embeddings

set -e  # Exit on error

echo "🔄 Regenerating embeddings..."
echo "   Model: sentence-transformers/all-MiniLM-L6-v2 (384 dims)"
echo "   Source: Wikipedia ML/AI articles (~8.3K sentences)"
echo "   Format: INT8 CVC compression"
echo ""

# Run setup function on Modal
modal run api/modal_api.py::setup_embeddings

echo ""
echo "✅ Done! Embeddings regenerated successfully"
echo "   Dimensions: 384 (all-MiniLM-L6-v2)"
echo "   Query and stored embeddings are now aligned"
echo ""
echo "Next: Redeploy the API with:"
echo "   modal deploy api/modal_api.py"
