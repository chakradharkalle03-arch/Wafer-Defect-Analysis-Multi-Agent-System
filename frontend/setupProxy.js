const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Proxy API requests to backend
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8001',
      changeOrigin: true,
      secure: false,
      logLevel: 'warn', // Reduced logging to avoid noise
      onError: (err, req, res) => {
        // Only log actual errors, not 404s for missing files
        if (err.code !== 'ECONNRESET' && err.code !== 'ECONNREFUSED') {
          console.error('Proxy error:', err.message);
        }
      }
    })
  );
  
  // Ignore manifest.json and other static files that don't need proxying
  app.use('/manifest.json', (req, res) => {
    res.status(404).json({ error: 'Not found' });
  });
};

