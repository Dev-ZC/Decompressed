import React, { useState } from 'react';
import CountUp from '../components/CountUp';
import PrismaticBurst from '../components/PrismaticBurst';
import PricingCalculator from '../components/PricingCalculator';
import '../assets/css/demo.css';

export default function Demo() {
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState(null);
  const [showBurst, setShowBurst] = useState(false);
  const [showStats, setShowStats] = useState(false);
  
  const mockResults = [
    { id: 1, text: 'Machine learning model optimization techniques', similarity: 0.94 },
    { id: 2, text: 'GPU acceleration for deep learning', similarity: 0.91 },
    { id: 3, text: 'Vector database compression methods', similarity: 0.88 },
    { id: 4, text: 'Efficient embedding storage strategies', similarity: 0.85 },
    { id: 5, text: 'Neural network inference optimization', similarity: 0.82 },
  ];
  
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    setShowBurst(true);
    setResults(null);
    
    // Simulate search delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setResults(mockResults);
    setLoading(false);
    setShowStats(true);
  };
  
  const handleBurstComplete = () => {
    setShowBurst(false);
  };
  
  return (
    <div className="demo">
      <div className="container">
        <h1>Vector Search Demo</h1>
        <p className="subtitle">Experience GPU-accelerated similarity search</p>
        
        <div className="demo-container">
          {/* Performance Comparison Stats */}
          <div className="performance-stats">
            <div className="stat-item stat-main">
              <div className="stat-value">
                {showStats ? <CountUp from={0} to={5.2} duration={1.5} /> : '0.0'}×
              </div>
              <div className="stat-label">GPU Decompression</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">
                {showStats ? <CountUp from={0} to={2.4} duration={1.5} delay={0.2} /> : '0.0'}×
              </div>
              <div className="stat-label">vs CPU</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">
                {showStats ? <CountUp from={0} to={0.4} duration={1.5} delay={0.4} /> : '0.0'}×
              </div>
              <div className="stat-label">Pure Python</div>
            </div>
          </div>

          {/* Search Input */}
          <form onSubmit={handleSearch} className="search-form">
            <div className="search-container">
              {showBurst && <PrismaticBurst onComplete={handleBurstComplete} />}
              <input
                type="text"
                className="search-input"
                placeholder="Enter your search query..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={loading}
              />
              <button type="submit" className="search-button" disabled={loading}>
                {loading ? (
                  <i className="fas fa-spinner fa-spin"></i>
                ) : (
                  <i className="fas fa-search"></i>
                )}
              </button>
            </div>
          </form>

          {/* Results */}
          {results && (
            <div className="search-results">
              <h3 className="results-title">Search Results</h3>
              {results.map((result, index) => (
                <div 
                  key={result.id} 
                  className="result-item"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="result-content">
                    <div className="result-text">{result.text}</div>
                    <div className="result-similarity">
                      {(result.similarity * 100).toFixed(0)}% match
                    </div>
                  </div>
                  <div className="result-bar">
                    <div 
                      className="result-bar-fill"
                      style={{ width: `${result.similarity * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Pricing Calculator */}
        <PricingCalculator />
        
        <div className="demo-explanation">
          <h2>How It Works</h2>
          <p>
            This demo simulates vector similarity search using GPU-accelerated decompression.
            Enter any query to find semantically similar embeddings from our compressed database.
          </p>
          <p>
            <strong>Performance:</strong> GPU decompression provides up to 5.2× faster search
            compared to CPU-based methods, enabling real-time similarity search at scale.
          </p>
          <p>
            <strong>ROI Calculator:</strong> Use the calculator above to estimate cost savings 
            based on your workload. Adjust GPU/CPU pricing, query volume, and time period to 
            see how much you could save by implementing GPU-accelerated decompression.
          </p>
        </div>
      </div>
    </div>
  );
}
