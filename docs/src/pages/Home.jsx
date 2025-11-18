import React from 'react';
import { Link } from 'react-router-dom';
import DecryptedText from '../components/DecryptedText';
import LightRays from '../components/LightRays';
import CodeBlock from '../components/CodeBlock';
import '../assets/css/home.css';

export default function Home() {
  return (
    <div className="home">
      {/* Hero Section with Light Rays Background */}
      <header className="hero">
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', overflow: 'hidden' }}>
          <LightRays
            raysOrigin="top-center"
            raysColor="#667eea"
            raysSpeed={0.8}
            lightSpread={0.6}
            rayLength={1.5}
            followMouse={true}
            mouseInfluence={0.15}
            noiseAmount={0.08}
            distortion={0.03}
            fadeDistance={1.2}
            saturation={0.7}
          />
        </div>
        <div className="hero-content">
          <h1 className="hero-title">
            <DecryptedText text="Decompressed" />
          </h1>
          <p className="hero-subtitle">
            GPU-native vector compression for embeddings and similarity search
          </p>
          <div className="cta-buttons">
            <Link to="/demo" className="btn btn-primary">
              <span>Try Live Demo</span>
              <i className="fas fa-arrow-right"></i>
            </Link>
            <a href="https://github.com/Dev-ZC/Decompressed" className="btn btn-secondary">
              <i className="fab fa-github"></i>
              <span>View on GitHub</span>
            </a>
          </div>
          <div className="hero-stats">
            <div className="stat">
              <div className="stat-value">5-10x</div>
              <div className="stat-label">Faster Loading</div>
            </div>
            <div className="stat">
              <div className="stat-value">2-4x</div>
              <div className="stat-label">Compression</div>
            </div>
            <div className="stat">
              <div className="stat-value">GPU</div>
              <div className="stat-label">Native</div>
            </div>
          </div>
        </div>
      </header>

      {/* Features */}
      <section id="features" className="section">
        <div className="container">
          <h2 className="section-title">Why Decompressed?</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <i className="fas fa-bolt feature-icon"></i>
              </div>
              <h3>Lightning Fast</h3>
              <p>Direct GPU decompression via CUDA/Triton kernels for 5-10x speedup</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <i className="fas fa-compress feature-icon"></i>
              </div>
              <h3>Highly Compressed</h3>
              <p>FP16 and INT8 compression with minimal accuracy loss</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <i className="fas fa-layer-group feature-icon"></i>
              </div>
              <h3>Smart Filtering</h3>
              <p>Section-based loading lets you access only the data you need</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <i className="fas fa-microchip feature-icon"></i>
              </div>
              <h3>Vendor Agnostic</h3>
              <p>Works on NVIDIA, AMD, and Intel GPUs via Triton</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <i className="fas fa-stream feature-icon"></i>
              </div>
              <h3>Streaming Support</h3>
              <p>Chunked format for efficient streaming of large datasets</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon-wrapper">
                <i className="fas fa-code feature-icon"></i>
              </div>
              <h3>Framework Agnostic</h3>
              <p>Works with NumPy, PyTorch, and CuPy out of the box</p>
            </div>
          </div>
        </div>
      </section>

      {/* Getting Started */}
      <section id="get-started" className="section">
        <div className="container">
          <h2 className="section-title">Get Started</h2>
          <CodeBlock
            code="pip install decompressed[gpu]"
            language="bash"
            title="Installation"
            icon="fas fa-terminal"
          />
          <CodeBlock
            code={`import numpy as np
from decompressed import pack_cvc, load_cvc

# Create and compress embeddings
embeddings = np.random.randn(100_000, 768).astype(np.float32)
pack_cvc(embeddings, "embeddings.cvc", compression="fp16")

# Load directly to GPU
vectors = load_cvc("embeddings.cvc", device="cuda")`}
            language="python"
            title="Quick Example"
            icon="fas fa-code"
          />
          <div className="cta-buttons">
            <a href="https://pypi.org/project/decompressed/" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
              <i className="fab fa-python"></i>
              <span>PyPI Package</span>
            </a>
            <a href="https://github.com/Dev-ZC/Decompressed#readme" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">
              <i className="fas fa-book"></i>
              <span>Documentation</span>
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}
