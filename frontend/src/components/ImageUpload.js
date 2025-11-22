import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './ImageUpload.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001/api/v1';

function ImageUpload({ onAnalysisStart, onAnalysisComplete, onError, loading }) {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [waferId, setWaferId] = useState('');
  const [batchId, setBatchId] = useState('');

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadedFile(file);

    // Start analysis
    onAnalysisStart();

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (waferId) formData.append('wafer_id', waferId);
      if (batchId) formData.append('batch_id', batchId);

      const response = await axios.post(
        `${API_BASE_URL}/analyze`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 300000, // 5 minutes timeout
        }
      );

      onAnalysisComplete(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      onError(
        error.response?.data?.detail || 
        error.message || 
        'Failed to analyze wafer image'
      );
    }
  }, [waferId, batchId, onAnalysisStart, onAnalysisComplete, onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    },
    maxFiles: 1,
    disabled: loading
  });

  return (
    <div className="card">
      <h2>Upload Wafer Image</h2>
      
      <div className="form-group">
        <label>Wafer ID (Optional)</label>
        <input
          type="text"
          value={waferId}
          onChange={(e) => setWaferId(e.target.value)}
          placeholder="Enter wafer ID"
          disabled={loading}
        />
      </div>

      <div className="form-group">
        <label>Batch ID (Optional)</label>
        <input
          type="text"
          value={batchId}
          onChange={(e) => setBatchId(e.target.value)}
          placeholder="Enter batch ID"
          disabled={loading}
        />
      </div>

      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${loading ? 'disabled' : ''}`}
      >
        <input {...getInputProps()} />
        {loading ? (
          <div className="upload-status">
            <div className="spinner"></div>
            <p>Analyzing wafer image...</p>
            <p className="status-text">This may take a few moments</p>
          </div>
        ) : isDragActive ? (
          <div className="upload-status">
            <p>Drop the image here...</p>
          </div>
        ) : (
          <div className="upload-status">
            <div className="upload-icon">📤</div>
            <p>Drag & drop a wafer image here, or click to select</p>
            <p className="upload-hint">Supports: JPG, PNG, TIFF</p>
            {uploadedFile && (
              <p className="uploaded-file">Selected: {uploadedFile.name}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageUpload;

