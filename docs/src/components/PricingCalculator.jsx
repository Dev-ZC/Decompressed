import React, { useState, useEffect } from 'react';
import './PricingCalculator.css';

export default function PricingCalculator() {
  const [gpuCostPerHour, setGpuCostPerHour] = useState(3.5);
  const [cpuCostPerHour, setCpuCostPerHour] = useState(0.8);
  const [numberOfUsers, setNumberOfUsers] = useState(10000);
  const [queriesPerUser, setQueriesPerUser] = useState(50);
  const [avgVectorsPerQuery, setAvgVectorsPerQuery] = useState(1000);
  const [storageGBCost, setStorageGBCost] = useState(0.15); // Per GB per month
  const [totalVectorsStored, setTotalVectorsStored] = useState(10000000); // 10M vectors
  const [period, setPeriod] = useState('month');

  const periodMultiplier = {
    day: 1,
    month: 30,
    year: 365
  };

  // Decompression throughput (vectors per second per instance)
  const gpuThroughput = 5.2e6; // 5.2M vectors/sec per GPU
  const cpuThroughput = 1.0e6; // 1M vectors/sec per CPU (realistic bottleneck)
  const pythonThroughput = 0.4e6; // 400K vectors/sec (pure Python, much slower)

  const queriesPerDay = (numberOfUsers * queriesPerUser) / 30; // Spread over a month
  const totalQueries = queriesPerDay * periodMultiplier[period];
  const totalVectors = totalQueries * avgVectorsPerQuery;

  // Storage calculations
  // Assume 768-dim vectors: FP32 = 3KB, INT8 = 768 bytes
  const vectorDims = 768;
  const fp32SizeBytes = vectorDims * 4; // 3072 bytes
  const int8SizeBytes = vectorDims * 1; // 768 bytes (4x compression)
  
  const fp32StorageGB = (totalVectorsStored * fp32SizeBytes) / (1024 ** 3);
  const int8StorageGB = (totalVectorsStored * int8SizeBytes) / (1024 ** 3);
  
  const storageCostFP32 = fp32StorageGB * storageGBCost * periodMultiplier[period];
  const storageCostINT8 = int8StorageGB * storageGBCost * periodMultiplier[period];
  const storageSavings = storageCostFP32 - storageCostINT8;

  // Calculate total processing time needed
  const totalProcessingTimeSeconds = totalVectors / 1e6; // Rough estimate for workload
  
  // Calculate required instances to handle the load
  // Assume we need to process within reasonable time (e.g., 24/7 serving)
  const hoursInPeriod = periodMultiplier[period] * 24;
  const secondsInPeriod = hoursInPeriod * 3600;
  
  // Calculate processing time per method
  const gpuProcessingSeconds = totalVectors / gpuThroughput;
  const cpuProcessingSeconds = totalVectors / cpuThroughput;
  const pythonProcessingSeconds = totalVectors / pythonThroughput;
  
  // Calculate utilization percentage (how much of the period we're processing)
  const gpuUtilization = Math.min(1, gpuProcessingSeconds / secondsInPeriod);
  const cpuUtilization = Math.min(1, cpuProcessingSeconds / secondsInPeriod);
  const pythonUtilization = Math.min(1, pythonProcessingSeconds / secondsInPeriod);
  
  // For realistic serving: Need to account for peak load handling
  // Typically need 2-3x capacity for peak hours and redundancy
  const loadMultiplier = 2.5; // Peak load factor
  
  // Calculate actual compute hours needed (considering always-on serving + peak capacity)
  const gpuComputeHours = Math.max(hoursInPeriod * 0.3, gpuProcessingSeconds / 3600 * loadMultiplier);
  const cpuComputeHours = Math.max(hoursInPeriod * 0.5, cpuProcessingSeconds / 3600 * loadMultiplier);
  const pythonComputeHours = Math.max(hoursInPeriod * 0.8, pythonProcessingSeconds / 3600 * loadMultiplier);

  // Calculate compute costs (realistic for 24/7 serving)
  const gpuComputeCost = gpuComputeHours * gpuCostPerHour;
  const cpuComputeCost = cpuComputeHours * cpuCostPerHour;
  const pythonComputeCost = pythonComputeHours * cpuCostPerHour;

  // Total costs (compute + storage)
  const gpuTotalCost = gpuComputeCost + storageCostINT8; // GPU uses INT8
  const cpuTotalCost = cpuComputeCost + storageCostFP32; // CPU uses FP32
  const pythonTotalCost = pythonComputeCost + storageCostFP32; // Python uses FP32

  // Savings
  const gpuVsCpuSavings = cpuTotalCost - gpuTotalCost;
  const gpuVsPythonSavings = pythonTotalCost - gpuTotalCost;
  
  // Time calculations for display
  const gpuTimeHours = gpuProcessingSeconds / 3600;
  const cpuTimeHours = cpuProcessingSeconds / 3600;
  const pythonTimeHours = pythonProcessingSeconds / 3600;

  const formatTime = (hours) => {
    if (hours < 1) {
      const minutes = hours * 60;
      if (minutes < 1) {
        return `${(minutes * 60).toFixed(1)}s`;
      }
      return `${minutes.toFixed(1)}m`;
    }
    if (hours > 24) {
      return `${(hours / 24).toFixed(1)}d`;
    }
    return `${hours.toFixed(1)}h`;
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  return (
    <div className="pricing-calculator">
      <h3 className="calculator-title">
        <i className="fas fa-calculator"></i>
        ROI Calculator [Approximation]
      </h3>
      
      <div className="calculator-inputs">
        <div className="input-row">
          <div className="input-group">
            <label>
              <span>Number of Users</span>
              <span className="input-value">{numberOfUsers.toLocaleString()}</span>
            </label>
            <input
              type="range"
              min="1000"
              max="1000000"
              step="1000"
              value={numberOfUsers}
              onChange={(e) => setNumberOfUsers(parseInt(e.target.value))}
              className="slider"
            />
          </div>

          <div className="input-group">
            <label>
              <span>Queries per User</span>
              <span className="input-value">{queriesPerUser}/month</span>
            </label>
            <input
              type="range"
              min="1"
              max="500"
              step="1"
              value={queriesPerUser}
              onChange={(e) => setQueriesPerUser(parseInt(e.target.value))}
              className="slider"
            />
          </div>
        </div>

        <div className="input-row">
          <div className="input-group">
            <label>
              <span>Vectors per Query</span>
              <span className="input-value">{avgVectorsPerQuery.toLocaleString()}</span>
            </label>
            <input
              type="range"
              min="100"
              max="10000"
              step="100"
              value={avgVectorsPerQuery}
              onChange={(e) => setAvgVectorsPerQuery(parseInt(e.target.value))}
              className="slider"
            />
          </div>

          <div className="input-group">
            <label>
              <span>Vectors in Database</span>
              <span className="input-value">{(totalVectorsStored / 1e6).toFixed(1)}M</span>
            </label>
            <input
              type="range"
              min="1000000"
              max="100000000"
              step="1000000"
              value={totalVectorsStored}
              onChange={(e) => setTotalVectorsStored(parseInt(e.target.value))}
              className="slider"
            />
          </div>
        </div>

        <div className="input-row">
          <div className="input-group">
            <label>
              <span>GPU Cost</span>
              <span className="input-value">${gpuCostPerHour.toFixed(2)}/hr</span>
            </label>
            <input
              type="range"
              min="0.5"
              max="10"
              step="0.1"
              value={gpuCostPerHour}
              onChange={(e) => setGpuCostPerHour(parseFloat(e.target.value))}
              className="slider"
            />
          </div>

          <div className="input-group">
            <label>
              <span>CPU Cost</span>
              <span className="input-value">${cpuCostPerHour.toFixed(2)}/hr</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={cpuCostPerHour}
              onChange={(e) => setCpuCostPerHour(parseFloat(e.target.value))}
              className="slider"
            />
          </div>
        </div>

        <div className="input-group">
          <label>
            <span>Storage Cost</span>
            <span className="input-value">${storageGBCost.toFixed(2)}/GB/month</span>
          </label>
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={storageGBCost}
            onChange={(e) => setStorageGBCost(parseFloat(e.target.value))}
            className="slider"
          />
        </div>

        <div className="input-group period-selector">
          <label>
            <span>Calculate for</span>
          </label>
          <div className="period-buttons">
            {['day', 'month', 'year'].map(p => (
              <button
                key={p}
                className={`period-btn ${period === p ? 'active' : ''}`}
                onClick={() => setPeriod(p)}
              >
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="calculator-results">
        <div className="result-comparison">
          <div className="comparison-item gpu">
            <div className="comparison-header">
              <i className="fas fa-microchip"></i>
              <span>GPU + INT8</span>
            </div>
            <div className="comparison-stats">
              <div className="stat">
                <span className="stat-label">Compute Time</span>
                <span className="stat-value">{formatTime(gpuTimeHours)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Compute Cost</span>
                <span className="stat-value">{formatCurrency(gpuComputeCost)}</span>
              </div>
            </div>
            <div className="storage-info">
              <div className="storage-stat">
                <span className="storage-label">Storage (INT8):</span>
                <span className="storage-value">{int8StorageGB.toFixed(1)} GB</span>
              </div>
              <div className="storage-stat">
                <span className="storage-label">Storage Cost:</span>
                <span className="storage-value">{formatCurrency(storageCostINT8)}</span>
              </div>
            </div>
            <div className="total-cost-display">
              <span className="total-label">Total Cost</span>
              <span className="total-value">{formatCurrency(gpuTotalCost)}</span>
            </div>
            <div className="comparison-badge best">Fastest & Most Efficient</div>
          </div>

          <div className="comparison-item cpu">
            <div className="comparison-header">
              <i className="fas fa-server"></i>
              <span>CPU + FP32</span>
            </div>
            <div className="comparison-stats">
              <div className="stat">
                <span className="stat-label">Compute Time</span>
                <span className="stat-value">{formatTime(cpuTimeHours)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Compute Cost</span>
                <span className="stat-value">{formatCurrency(cpuComputeCost)}</span>
              </div>
            </div>
            <div className="storage-info">
              <div className="storage-stat">
                <span className="storage-label">Storage (FP32):</span>
                <span className="storage-value">{fp32StorageGB.toFixed(1)} GB</span>
              </div>
              <div className="storage-stat">
                <span className="storage-label">Storage Cost:</span>
                <span className="storage-value">{formatCurrency(storageCostFP32)}</span>
              </div>
            </div>
            <div className="total-cost-display">
              <span className="total-label">Total Cost</span>
              <span className="total-value">{formatCurrency(cpuTotalCost)}</span>
            </div>
            <div className="comparison-savings">
              {gpuVsCpuSavings > 0 ? (
                <span className="savings-positive">
                  Save {formatCurrency(gpuVsCpuSavings)} with GPU + INT8
                </span>
              ) : (
                <span className="savings-negative">
                  {formatCurrency(Math.abs(gpuVsCpuSavings))} more expensive
                </span>
              )}
            </div>
          </div>

          <div className="comparison-item python">
            <div className="comparison-header">
              <i className="fab fa-python"></i>
              <span>Pure Python</span>
            </div>
            <div className="comparison-stats">
              <div className="stat">
                <span className="stat-label">Compute Time</span>
                <span className="stat-value">{formatTime(pythonTimeHours)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Compute Cost</span>
                <span className="stat-value">{formatCurrency(pythonComputeCost)}</span>
              </div>
            </div>
            <div className="storage-info">
              <div className="storage-stat">
                <span className="storage-label">Storage (FP32):</span>
                <span className="storage-value">{fp32StorageGB.toFixed(1)} GB</span>
              </div>
              <div className="storage-stat">
                <span className="storage-label">Storage Cost:</span>
                <span className="storage-value">{formatCurrency(storageCostFP32)}</span>
              </div>
            </div>
            <div className="total-cost-display">
              <span className="total-label">Total Cost</span>
              <span className="total-value">{formatCurrency(pythonTotalCost)}</span>
            </div>
            <div className="comparison-savings">
              {gpuVsPythonSavings > 0 ? (
                <span className="savings-positive">
                  Save {formatCurrency(gpuVsPythonSavings)} with GPU + INT8
                </span>
              ) : (
                <span className="savings-negative">
                  {formatCurrency(Math.abs(gpuVsPythonSavings))} more expensive
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="total-summary">
          <div className="summary-item">
            <i className="fas fa-database"></i>
            <div>
              <span className="summary-label">Storage Savings (INT8 vs FP32)</span>
              <span className="summary-value">{formatCurrency(storageSavings)}/{period}</span>
            </div>
          </div>
          <div className="summary-item">
            <i className="fas fa-compress"></i>
            <div>
              <span className="summary-label">Storage Reduced by</span>
              <span className="summary-value">75% ({fp32StorageGB.toFixed(0)} → {int8StorageGB.toFixed(0)} GB)</span>
            </div>
          </div>
          <div className="summary-item highlight">
            <i className="fas fa-piggy-bank"></i>
            <div>
              <span className="summary-label">Total Savings per {period}</span>
              <span className="summary-value">{formatCurrency(Math.max(gpuVsCpuSavings, gpuVsPythonSavings))}</span>
            </div>
          </div>
        </div>

        {/* Calculation Breakdown */}
        <div className="calculation-breakdown">
          <h4 className="breakdown-title">
            <i className="fas fa-info-circle"></i>
            How These Savings Are Calculated
          </h4>
          
          <div className="breakdown-section">
            <h5>Query Volume</h5>
            <div className="breakdown-item">
              <span className="breakdown-label">Total Users:</span>
              <span className="breakdown-value">{numberOfUsers.toLocaleString()}</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">Queries per User per Month:</span>
              <span className="breakdown-value">{queriesPerUser}</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">Total Queries per {period}:</span>
              <span className="breakdown-value">{totalQueries.toLocaleString()} queries</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">Vectors per Query:</span>
              <span className="breakdown-value">{avgVectorsPerQuery.toLocaleString()} vectors</span>
            </div>
            <div className="breakdown-item highlight">
              <span className="breakdown-label">Total Vectors Processed:</span>
              <span className="breakdown-value">
                {totalVectors >= 1e9 
                  ? `${(totalVectors / 1e9).toFixed(2)}B vectors` 
                  : `${(totalVectors / 1e6).toFixed(2)}M vectors`}
                {' '}({totalQueries.toLocaleString()} × {avgVectorsPerQuery.toLocaleString()})
              </span>
            </div>
          </div>

          <div className="breakdown-section">
            <h5>Decompression Performance</h5>
            <div className="breakdown-grid">
              <div>
                <div className="breakdown-item">
                  <span className="breakdown-label">GPU Throughput:</span>
                  <span className="breakdown-value">{(gpuThroughput / 1e6).toFixed(1)}M vectors/sec</span>
                </div>
                <div className="breakdown-item">
                  <span className="breakdown-label">Pure Processing Time:</span>
                  <span className="breakdown-value">{formatTime(gpuTimeHours)}</span>
                </div>
              </div>
              <div>
                <div className="breakdown-item">
                  <span className="breakdown-label">CPU Throughput:</span>
                  <span className="breakdown-value">{(cpuThroughput / 1e6).toFixed(1)}M vectors/sec</span>
                </div>
                <div className="breakdown-item">
                  <span className="breakdown-label">Pure Processing Time:</span>
                  <span className="breakdown-value">{formatTime(cpuTimeHours)}</span>
                </div>
              </div>
              <div>
                <div className="breakdown-item">
                  <span className="breakdown-label">Python Throughput:</span>
                  <span className="breakdown-value">{(pythonThroughput / 1e6).toFixed(1)}M vectors/sec</span>
                </div>
                <div className="breakdown-item">
                  <span className="breakdown-label">Pure Processing Time:</span>
                  <span className="breakdown-value">{formatTime(pythonTimeHours)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="breakdown-section">
            <h5>Compute Cost Modeling</h5>
            <p className="breakdown-explanation">
              <strong>Why compute costs matter:</strong> In production, you need 24/7 serving capacity with 
              redundancy for peak loads. Slower decompression means:
            </p>
            <ul className="breakdown-list">
              <li>More instances running continuously to handle throughput</li>
              <li>2-3× capacity overhead for peak hours and failover</li>
              <li>CPU: {((cpuUtilization * 100).toFixed(0))}% utilization requires {(cpuComputeHours / hoursInPeriod * 100).toFixed(0)}% uptime</li>
              <li>GPU: {((gpuUtilization * 100).toFixed(0))}% utilization requires {(gpuComputeHours / hoursInPeriod * 100).toFixed(0)}% uptime</li>
            </ul>
            <div className="breakdown-item highlight">
              <span className="breakdown-label">GPU Compute Hours ({period}):</span>
              <span className="breakdown-value">{gpuComputeHours.toFixed(0)} hrs @ ${gpuCostPerHour}/hr = {formatCurrency(gpuComputeCost)}</span>
            </div>
            <div className="breakdown-item highlight">
              <span className="breakdown-label">CPU Compute Hours ({period}):</span>
              <span className="breakdown-value">{cpuComputeHours.toFixed(0)} hrs @ ${cpuCostPerHour}/hr = {formatCurrency(cpuComputeCost)}</span>
            </div>
          </div>

          <div className="breakdown-section">
            <h5>Storage Cost Breakdown</h5>
            <div className="breakdown-item">
              <span className="breakdown-label">Vectors in Database:</span>
              <span className="breakdown-value">{(totalVectorsStored / 1e6).toFixed(1)}M vectors @ 768 dimensions</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">FP32 Storage (4 bytes/dim):</span>
              <span className="breakdown-value">{fp32StorageGB.toFixed(1)} GB × ${storageGBCost}/GB = {formatCurrency(storageCostFP32)}/{period}</span>
            </div>
            <div className="breakdown-item">
              <span className="breakdown-label">INT8 Storage (1 byte/dim):</span>
              <span className="breakdown-value">{int8StorageGB.toFixed(1)} GB × ${storageGBCost}/GB = {formatCurrency(storageCostINT8)}/{period}</span>
            </div>
            <div className="breakdown-item highlight">
              <span className="breakdown-label">Storage Savings (75% reduction):</span>
              <span className="breakdown-value">{formatCurrency(storageSavings)}/{period}</span>
            </div>
          </div>

          <div className="breakdown-section final">
            <h5>Total Cost Comparison</h5>
            <div className="breakdown-comparison">
              <div className="breakdown-method">
                <strong>GPU + INT8:</strong>
                <div>{formatCurrency(gpuComputeCost)} compute + {formatCurrency(storageCostINT8)} storage = <strong>{formatCurrency(gpuTotalCost)}</strong></div>
              </div>
              <div className="breakdown-method">
                <strong>CPU + FP32:</strong>
                <div>{formatCurrency(cpuComputeCost)} compute + {formatCurrency(storageCostFP32)} storage = <strong>{formatCurrency(cpuTotalCost)}</strong></div>
              </div>
              <div className="breakdown-savings-box">
                <i className="fas fa-arrow-down"></i>
                <span>You save {formatCurrency(gpuVsCpuSavings)} per {period} with GPU + INT8</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
