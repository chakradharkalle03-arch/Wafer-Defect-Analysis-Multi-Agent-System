import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import AnalysisResults from './components/AnalysisResults';
import Dashboard from './components/Dashboard';

function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalysisComplete = (result) => {
    setAnalysisResult(result);
    setLoading(false);
    setError(null);
  };

  const handleAnalysisStart = () => {
    setLoading(true);
    setError(null);
    setAnalysisResult(null);
  };

  const handleError = (err) => {
    setError(err);
    setLoading(false);
  };

  return (
    <div className="App">
      <Header />
      <div className="container">
        <Dashboard />
        <ImageUpload
          onAnalysisStart={handleAnalysisStart}
          onAnalysisComplete={handleAnalysisComplete}
          onError={handleError}
          loading={loading}
        />
        {error && (
          <div className="error-message">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}
        {analysisResult && (
          <AnalysisResults result={analysisResult} />
        )}
      </div>
    </div>
  );
}

export default App;

