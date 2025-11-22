import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001/api/v1';

function Dashboard() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setHealth(response.data);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealth({ status: 'unhealthy' });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="card dashboard">
        <h2>System Status</h2>
        <div className="loading-spinner">
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="card dashboard">
      <h2>System Dashboard</h2>
      <div className="dashboard-content">
        <div className="status-indicator">
          <div className={`status-dot ${health?.status === 'healthy' ? 'healthy' : 'unhealthy'}`}></div>
          <span className="status-text">
            System Status: <strong>{health?.status?.toUpperCase() || 'UNKNOWN'}</strong>
          </span>
        </div>

        {health?.agents_ready && (
          <div className="agents-status">
            <h3>Agents Status</h3>
            <div className="agents-grid">
              {Object.entries(health.agents_ready).map(([agent, ready]) => (
                <div key={agent} className="agent-item">
                  <span className={`agent-status ${ready ? 'ready' : 'not-ready'}`}>
                    {ready ? '✓' : '✗'}
                  </span>
                  <span className="agent-name">{agent.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {health?.models_loaded && (
          <div className="models-status">
            <h3>Models Status</h3>
            <div className="models-grid">
              {Object.entries(health.models_loaded)
                .filter(([model]) => !model.includes('yolo') && !model.includes('vit_local'))
                .map(([model, loaded]) => {
                  // Format model name for display
                  let displayName = model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                  if (model.includes('huggingface')) {
                    displayName = displayName.replace('Huggingface', 'HF').replace('Detr', 'DETR').replace('Vit', 'ViT');
                  }
                  return (
                    <div key={model} className="model-item">
                      <span className={`model-status ${loaded ? 'loaded' : 'not-loaded'}`}>
                        {loaded ? '✓' : '✗'}
                      </span>
                      <span className="model-name">{displayName}</span>
                    </div>
                  );
                })}
            </div>
            {health.models_loaded.huggingface_detr && health.models_loaded.huggingface_vit && (
              <div className="model-info" style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
                <span>✓ Using HuggingFace Inference API (Open Source Models)</span>
              </div>
            )}
          </div>
        )}

        <div className="system-info">
          <p><strong>Version:</strong> {health?.version || '1.0.0'}</p>
          <p><strong>API Endpoint:</strong> {API_BASE_URL}</p>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;

