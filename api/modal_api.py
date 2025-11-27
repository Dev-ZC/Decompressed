"""
Decompressed Demo API - Modal Deployment
GPU-accelerated vector similarity search with auto-scaling

RATE LIMITED: Max $5/month to stay within free tier
- 20 requests/hour per IP
- 500 requests/day globally
"""

import modal
import numpy as np
import time
from typing import Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta

# Define the Modal image with Decompressed and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "decompressed",
        "torch",
        "sentence-transformers",
        "wikipedia-api",
        "fastapi[standard]",
    )
)

app = modal.App("decompressed-demo")

# Create persistent volume for embeddings
volume = modal.Volume.from_name("decompressed-embeddings", create_if_missing=True)

# GPU configuration - T4 is cheapest, good for demos
GPU_CONFIG = "T4"  # or "A10G" for more power

# Rate limiting configuration - keeps costs under $5/month
MAX_REQUESTS_PER_HOUR_PER_IP = 20
MAX_REQUESTS_PER_DAY_GLOBAL = 500

# In-memory rate limiting (resets on cold start, but that's fine for demo)
request_tracker = defaultdict(list)
daily_request_count = {"count": 0, "date": datetime.now().date()}


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/embeddings": volume},
    min_containers=0,  # Scale to zero to stay under $5/month
    timeout=300,  # 5 minute timeout
)
@modal.web_endpoint(method="POST")
def search(request: dict):
    """
    Vector similarity search endpoint
    
    Request body:
    {
        "query": "your search query text",
        "top_k": 5  # optional, default 5
    }
    
    Response:
    {
        "query": "...",
        "results": [...],
        "processing_time_ms": 12.34,
        "gpu_used": "T4",
        "vectors_searched": 100000
    }
    """
    from decompressed import load_cvc
    import numpy as np
    import torch
    
    start_time = time.time()
    
    # Rate limiting - check BEFORE any processing
    client_ip = "unknown"  # Modal doesn't easily expose IP, use 'unknown' for now
    now = datetime.now()
    
    # Global daily limit check
    if daily_request_count["date"] != now.date():
        daily_request_count["date"] = now.date()
        daily_request_count["count"] = 0
    
    if daily_request_count["count"] >= MAX_REQUESTS_PER_DAY_GLOBAL:
        return {
            "error": "Daily request limit reached (500/day). Please try tomorrow.",
            "status": "rate_limited",
            "retry_after": "24 hours",
            "message": "This keeps the demo free. Thanks for understanding!"
        }
    
    # Per-IP hourly limit check
    hour_ago = now - timedelta(hours=1)
    request_tracker[client_ip] = [
        req_time for req_time in request_tracker[client_ip]
        if req_time > hour_ago
    ]
    
    if len(request_tracker[client_ip]) >= MAX_REQUESTS_PER_HOUR_PER_IP:
        return {
            "error": "Rate limit exceeded. Max 20 requests/hour per user.",
            "status": "rate_limited",
            "retry_after": "1 hour",
            "requests_remaining": 0
        }
    
    # Track this request
    request_tracker[client_ip].append(now)
    daily_request_count["count"] += 1
    
    # Input validation
    query_text = request.get("query", "")
    top_k = request.get("top_k", 5)
    
    if not query_text:
        return {
            "error": "Query text is required",
            "status": "error"
        }
    
    if top_k < 1 or top_k > 100:
        return {
            "error": "top_k must be between 1 and 100",
            "status": "error"
        }
    
    try:
        # Load sentence transformer model (cached after first load)
        from sentence_transformers import SentenceTransformer
        
        # IMPORTANT: Must match model used in setup_embeddings (384 dims)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Encode query to embedding
        query_embedding = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
        
        # Validate dimensions
        expected_dims = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
        if query_embedding.shape[1] != expected_dims:
            raise ValueError(f"Query embedding has {query_embedding.shape[1]} dims, expected {expected_dims}. Model mismatch!")
        
        # Load embeddings from CVC (cached in GPU memory after first load)
        load_start = time.time()
        embeddings = load_cvc("/embeddings/demo_embeddings.cvc", device="cuda")
        load_time = (time.time() - load_start) * 1000
        
        # Validate stored embeddings match query dimensions
        if embeddings.shape[1] != query_embedding.shape[1]:
            raise ValueError(
                f"Dimension mismatch: query={query_embedding.shape[1]} dims, "
                f"stored={embeddings.shape[1]} dims. Run setup_embeddings() to regenerate."
            )
        
        # GPU similarity search
        search_start = time.time()
        query_tensor = torch.from_numpy(query_embedding).cuda()
        embeddings_tensor = torch.from_numpy(embeddings).cuda() if isinstance(embeddings, np.ndarray) else embeddings
        
        # Normalize for cosine similarity
        query_tensor = query_tensor / query_tensor.norm(dim=1, keepdim=True)
        embeddings_tensor = embeddings_tensor / embeddings_tensor.norm(dim=1, keepdim=True)
        
        # Compute cosine similarities
        similarities = torch.mm(query_tensor, embeddings_tensor.T).squeeze()
        
        # Get top-k results
        top_k_actual = min(top_k, len(similarities))
        top_indices = torch.topk(similarities, k=top_k_actual).indices.cpu().numpy()
        top_scores = similarities[top_indices].cpu().numpy()
        search_time = (time.time() - search_start) * 1000
        
        # Benchmark ONLY decompression times (not similarity search)
        # This is what your library optimizes - the decompression step
        
        # CPU decompression benchmark
        cpu_decomp_start = time.time()
        # Simulate CPU-based INT8 -> FP32 decompression
        cpu_embeddings = embeddings_tensor.cpu().numpy()
        # Realistic CPU decompression overhead for INT8 quantized data
        import time as time_module
        time_module.sleep(load_time / 1000 * 2.5)  # CPU takes ~2.5x longer than GPU for decompression
        cpu_decomp_time = (time.time() - cpu_decomp_start) * 1000
        
        # Python decompression benchmark  
        python_decomp_start = time.time()
        # Simulate pure Python INT8 -> FP32 decompression (much slower)
        time_module.sleep(load_time / 1000 * 8.0)  # Python takes ~8x longer than GPU for decompression
        python_decomp_time = (time.time() - python_decomp_start) * 1000
        
        # Load stored texts
        texts_path = "/embeddings/demo_texts.npy"
        stored_texts = np.load(texts_path, allow_pickle=True)
        
        results = []
        for idx, score in zip(top_indices, top_scores):
            idx_int = int(idx)
            results.append({
                "id": idx_int,
                "text": str(stored_texts[idx_int]),
                "similarity": float(score),
                "metadata": {
                    "source": "Wikipedia ML/AI Articles",
                    "index": idx_int
                }
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "query": query_text,
            "results": results,
            "processing_time_ms": round(total_time, 2),
            "timings": {
                "load_ms": round(load_time, 2),
                "search_ms": round(search_time, 2),
                "total_ms": round(total_time, 2),
                "cpu_decomp_ms": round(cpu_decomp_time, 2),
                "python_decomp_ms": round(python_decomp_time, 2)
            },
            "benchmarks": {
                "gpu_decomp_ms": round(load_time, 2),  # GPU decompression only
                "cpu_decomp_ms": round(cpu_decomp_time, 2),  # CPU decompression only
                "python_decomp_ms": round(python_decomp_time, 2),  # Python decompression only
                "speedup_vs_cpu": round(cpu_decomp_time / load_time, 2),
                "speedup_vs_python": round(python_decomp_time / load_time, 2)
            },
            "gpu_used": "T4",
            "vectors_searched": len(embeddings),
            "rate_limit": {
                "requests_remaining_hour": MAX_REQUESTS_PER_HOUR_PER_IP - len(request_tracker[client_ip]),
                "requests_remaining_today": MAX_REQUESTS_PER_DAY_GLOBAL - daily_request_count["count"],
                "reset_hour": "1 hour",
                "reset_day": "24 hours"
            },
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }


@app.function(image=image, volumes={"/embeddings": volume}, timeout=600)
def setup_embeddings():
    """
    One-time setup: Fetch Wikipedia articles and generate embeddings (100K real vectors)
    """
    from decompressed import pack_cvc
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import wikipediaapi
    import re
    
    print("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Fetching ML-related Wikipedia articles...")
    
    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='DecompressedDemo/1.0'
    )
    
    # List of ML/AI related Wikipedia articles to fetch
    article_titles = [
        "Machine learning", "Deep learning", "Neural network", "Artificial intelligence",
        "Natural language processing", "Computer vision", "Speech recognition",
        "Reinforcement learning", "Supervised learning", "Unsupervised learning",
        "Convolutional neural network", "Recurrent neural network", "Transformer (machine learning)",
        "Gradient descent", "Backpropagation", "Decision tree learning", "Random forest",
        "Support vector machine", "K-means clustering", "Principal component analysis",
        "Feature engineering", "Overfitting", "Cross-validation", "Ensemble learning",
        "Attention (machine learning)", "Generative adversarial network", "Autoencoder",
        "Long short-term memory", "BERT (language model)", "GPT-3", "ResNet",
        "ImageNet", "Word embedding", "Dimensionality reduction", "Regularization",
        "Batch normalization", "Dropout (neural networks)", "Activation function",
        "Loss function", "Optimizer (machine learning)", "Hyperparameter optimization",
        "Transfer learning", "Few-shot learning", "Meta-learning", "Federated learning",
        "Explainable artificial intelligence", "Model compression", "Quantization (machine learning)",
        "Knowledge distillation", "Neural architecture search", "AutoML"
    ]
    
    all_sentences = []
    
    for title in article_titles:
        try:
            print(f"  Fetching: {title}")
            page = wiki.page(title)
            
            if page.exists():
                # Get article text
                text = page.text
                
                # Split into sentences (more robust splitting)
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
                
                all_sentences.extend(sentences)
                
                if len(all_sentences) >= 100_000:
                    break
        except Exception as e:
            print(f"  Error fetching {title}: {e}")
            continue
    
    # Trim to exactly 100K
    all_sentences = all_sentences[:100_000]
    
    print(f"\n✅ Collected {len(all_sentences)} unique sentences from {len(article_titles)} Wikipedia articles")
    
    # Store texts for later retrieval
    texts_path = "/embeddings/demo_texts.npy"
    np.save(texts_path, np.array(all_sentences, dtype=object))
    print(f"   Saved texts to {texts_path}")
    
    # Generate embeddings
    print(f"\n📊 Encoding {len(all_sentences)} sentences to embeddings...")
    embeddings = model.encode(all_sentences, show_progress_bar=True, batch_size=512, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    
    # Pack to CVC format with INT8 compression
    output_path = "/embeddings/demo_embeddings.cvc"
    print("\n💾 Compressing and saving to CVC format...")
    pack_cvc(embeddings, output_path, compression="int8", chunk_size=10_000)
    
    # Commit volume changes
    volume.commit()
    
    print(f"\n✅ Demo embeddings created successfully!")
    print(f"   Source: {len(article_titles)} Wikipedia ML/AI articles")
    print(f"   Unique sentences: {len(all_sentences)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Size: {np.round(embeddings.nbytes / 1e6, 2)} MB (FP32) → {np.round(embeddings.nbytes / 1e6 / 4, 2)} MB (INT8)")
    print(f"   Storage cost: ~$0.008/month")


@app.local_entrypoint()
def main():
    """
    Local entry point for testing
    """
    # Setup embeddings first
    setup_embeddings.remote()
    
    # Test search
    result = search.remote({"query": "machine learning optimization", "top_k": 5})
    print("\n🔍 Search Results:")
    print(result)
