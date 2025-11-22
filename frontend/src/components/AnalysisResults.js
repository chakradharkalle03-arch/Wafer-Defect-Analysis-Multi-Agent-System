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
      </div>
    </div>
  );
}

export default AnalysisResults;

