import React, { useState } from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './AnalysisResults.css';

function AnalysisResults({ result }) {
  const [activeTab, setActiveTab] = useState('overview');

  // Prepare data for charts
  const defectTypeData = Object.entries(result.defect_summary).map(([name, value]) => ({
    name: name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value
  }));

  const processStepData = result.root_causes.reduce((acc, rc) => {
    const step = rc.process_step;
    acc[step] = (acc[step] || 0) + 1;
    return acc;
  }, {});

  const processStepChartData = Object.entries(processStepData).map(([name, value]) => ({
    name,
    count: value
  }));

  const confidenceData = result.classifications.map(c => ({
    name: c.defect_id,
    confidence: (c.confidence * 100).toFixed(1)
  }));

  const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140'];

  const getSeverityBadge = (score) => {
    if (score > 0.8) return { class: 'badge-danger', text: 'CRITICAL' };
    if (score > 0.5) return { class: 'badge-warning', text: 'HIGH' };
    if (score > 0.3) return { class: 'badge-info', text: 'MODERATE' };
    return { class: 'badge-success', text: 'LOW' };
  };

  const severity = getSeverityBadge(result.severity_score);

  return (
    <div className="card analysis-results">
      <h2>Analysis Results</h2>
      
      {/* Tabs */}
      <div className="tabs">
        <button
          className={activeTab === 'overview' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={activeTab === 'defects' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('defects')}
        >
          Defects ({result.total_defects})
        </button>
        <button
          className={activeTab === 'root-causes' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('root-causes')}
        >
          Root Causes
        </button>
        <button
          className={activeTab === 'charts' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('charts')}
        >
          Analytics
        </button>
        {result.defect_map && (
          <button
            className={activeTab === 'map' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('map')}
          >
            Defect Map
          </button>
        )}
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview">
            <div className="grid">
              <div className="stat-box">
                <h3>Total Defects</h3>
                <div className="value">{result.total_defects}</div>
              </div>
              <div className="stat-box">
                <h3>Severity Score</h3>
                <div className="value">{(result.severity_score * 100).toFixed(1)}%</div>
                <span className={`badge ${severity.class}`}>{severity.text}</span>
              </div>
              <div className="stat-box">
                <h3>Defect Types</h3>
                <div className="value">{Object.keys(result.defect_summary).length}</div>
              </div>
              <div className="stat-box">
                <h3>Analysis ID</h3>
                <div className="value" style={{ fontSize: '14px' }}>{result.analysis_id}</div>
              </div>
            </div>

            <div className="defect-summary">
              <h3>Defect Type Summary</h3>
              <div className="defect-list">
                {Object.entries(result.defect_summary).map(([type, count]) => (
                  <div key={type} className="defect-item">
                    <span className="defect-type">{type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                    <span className="defect-count">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {result.report_path && (
              <div className="report-download">
                <a
                  href={`${process.env.REACT_APP_API_URL || 'http://localhost:8001/api/v1'}/report/${result.analysis_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="button"
                >
                  📄 Download Full Report (PDF)
                </a>
              </div>
            )}
          </div>
        )}

        {activeTab === 'defects' && (
          <div className="defects-list">
            {result.defects.map((defect, index) => {
              const classification = result.classifications[index];
              
              return (
                <div key={defect.defect_id} className="defect-card">
                  <div className="defect-header">
                    <h4>{defect.defect_id}</h4>
                    <span className={`badge badge-info`}>
                      {classification.defect_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </span>
                  </div>
                  <div className="defect-details">
                    <div className="detail-row">
                      <span className="detail-label">Confidence:</span>
                      <span className="detail-value">{(classification.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Location:</span>
                      <span className="detail-value">
                        ({defect.bbox.x_min.toFixed(1)}, {defect.bbox.y_min.toFixed(1)}) - 
                        ({defect.bbox.x_max.toFixed(1)}, {defect.bbox.y_max.toFixed(1)})
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Area:</span>
                      <span className="detail-value">{defect.area.toFixed(2)} pixels²</span>
                    </div>
                    {classification.description && (
                      <div className="detail-row">
                        <span className="detail-label">Description:</span>
                        <span className="detail-value">{classification.description}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === 'root-causes' && (
          <div className="root-causes-list">
            {result.root_causes.map((rc, index) => (
              <div key={rc.defect_id} className="root-cause-card">
                <div className="root-cause-header">
                  <h4>Defect: {rc.defect_id}</h4>
                  <span className="badge badge-warning">{rc.process_step}</span>
                </div>
                <div className="root-cause-details">
                  <div className="detail-row">
                    <span className="detail-label">Likely Cause:</span>
                    <span className="detail-value">{rc.likely_cause}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Confidence:</span>
                    <span className="detail-value">{(rc.confidence * 100).toFixed(1)}%</span>
                  </div>
                  {rc.recommendations && rc.recommendations.length > 0 && (
                    <div className="recommendations">
                      <h5>Recommendations:</h5>
                      <ul>
                        {rc.recommendations.map((rec, i) => (
                          <li key={i}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'charts' && (
          <div className="charts">
            <div className="chart-container">
              <h3>Defect Type Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={defectTypeData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {defectTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h3>Defects by Process Step</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={processStepChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#667eea" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h3>Classification Confidence</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={confidenceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="confidence" fill="#764ba2" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'map' && result.defect_map && (
          <div className="defect-map">
            <h3>Wafer Defect Spatial Map</h3>
            
            {/* Map Image */}
            {result.defect_map.map_image_path && (
              <div className="map-image-container">
                <img 
                  src={`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/v1/map/${result.defect_map.map_image_path.split('/').pop()}`}
                  alt="Defect Map"
                  className="defect-map-image"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling && (e.target.nextSibling.style.display = 'block');
                  }}
                />
                <div style={{display: 'none', padding: '20px', color: '#666'}}>
                  Map image not available. Path: {result.defect_map.map_image_path}
                </div>
              </div>
            )}

            {/* Spatial Statistics */}
            {result.defect_map.spatial_statistics && (
              <div className="spatial-statistics">
                <h4>Spatial Analysis</h4>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Clusters Found:</span>
                    <span className="stat-value">{result.defect_map.spatial_statistics.num_clusters || 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Defect Density:</span>
                    <span className="stat-value">
                      {(result.defect_map.spatial_statistics.defect_density || 0).toFixed(4)} defects/pixel²
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Mean Distance from Centroid:</span>
                    <span className="stat-value">
                      {(result.defect_map.spatial_statistics.mean_distance_from_centroid || 0).toFixed(1)} pixels
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Spatial Uniformity:</span>
                    <span className="stat-value">
                      {(result.defect_map.spatial_statistics.spatial_uniformity || 0).toFixed(3)}
                    </span>
                  </div>
                  {result.defect_map.spatial_statistics.centroid && (
                    <>
                      <div className="stat-item">
                        <span className="stat-label">Centroid X:</span>
                        <span className="stat-value">
                          {result.defect_map.spatial_statistics.centroid.x.toFixed(1)}
                        </span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Centroid Y:</span>
                        <span className="stat-value">
                          {result.defect_map.spatial_statistics.centroid.y.toFixed(1)}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Clusters */}
            {result.defect_map.clusters && result.defect_map.clusters.length > 0 && (
              <div className="clusters-section">
                <h4>Defect Clusters ({result.defect_map.clusters.length})</h4>
                <div className="clusters-list">
                  {result.defect_map.clusters.map((cluster) => (
                    <div key={cluster.cluster_id} className="cluster-card">
                      <div className="cluster-header">
                        <h5>{cluster.cluster_id}</h5>
                        <span className="badge badge-info">{cluster.size} defects</span>
                      </div>
                      <div className="cluster-defects">
                        {cluster.defects.slice(0, 5).map((defect) => (
                          <div key={defect.defect_id} className="cluster-defect-item">
                            <span>{defect.defect_id}</span>
                            <span className="defect-type-badge">{defect.type}</span>
                          </div>
                        ))}
                        {cluster.defects.length > 5 && (
                          <div className="more-defects">+{cluster.defects.length - 5} more</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Defect Positions */}
            {result.defect_map.defect_positions && result.defect_map.defect_positions.length > 0 && (
              <div className="defect-positions">
                <h4>Defect Positions</h4>
                <div className="positions-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Defect ID</th>
                        <th>Type</th>
                        <th>Center (X, Y)</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.defect_map.defect_positions.slice(0, 20).map((pos) => (
                        <tr key={pos.defect_id}>
                          <td>{pos.defect_id}</td>
                          <td>{pos.type.replace('_', ' ')}</td>
                          <td>({pos.center[0].toFixed(1)}, {pos.center[1].toFixed(1)})</td>
                          <td>{(pos.confidence * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {result.defect_map.defect_positions.length > 20 && (
                    <div className="more-positions">
                      Showing first 20 of {result.defect_map.defect_positions.length} defects
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalysisResults;

