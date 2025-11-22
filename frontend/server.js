/**
 * Simple Node.js server for React app
 * Alternative to webpack dev server to avoid allowedHosts issues
 */
const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3002;

// Proxy API requests to backend
app.use(
  '/api',
  createProxyMiddleware({
    target: 'http://localhost:8001',
    changeOrigin: true,
    secure: false,
  })
);

// Serve static files from build directory (if built)
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, 'build')));
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'build', 'index.html'));
  });
} else {
  // Development mode - use webpack dev server or serve from public
  app.use(express.static(path.join(__dirname, 'public')));
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
  });
}

app.listen(PORT, 'localhost', () => {
  console.log(`\n=== Frontend Server Running ===`);
  console.log(`Port: ${PORT}`);
  console.log(`URL: http://localhost:${PORT}`);
  console.log(`API Proxy: http://localhost:8001/api\n`);
});

