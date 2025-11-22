const path = require('path');

module.exports = function override(config, env) {
  // DEEP FIX for webpack dev server allowedHosts issue
  // The problem: react-scripts 5.0.1 sets allowedHosts internally with empty strings
  // Solution: Intercept and fix it at multiple levels
  
  if (config.devServer) {
    // Method 1: Set allowedHosts BEFORE any validation
    // Use a valid array with non-empty strings
    config.devServer.allowedHosts = ['localhost', '127.0.0.1', '.localhost'];
    
    // Method 2: Also set disableHostCheck (older webpack compatibility)
    config.devServer.disableHostCheck = true;
    
    // Method 3: Explicitly set host and port
    config.devServer.host = 'localhost';
    config.devServer.port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3002;
    
    // Method 4: Disable client overlay for warnings
    config.devServer.client = {
      ...(config.devServer.client || {}),
      overlay: {
        warnings: false,
        errors: true
      },
      logging: 'warn'
    };
    
    // Method 5: Set onHeaders to ensure allowedHosts is never empty
    const originalOnHeaders = config.devServer.onHeaders;
    config.devServer.onHeaders = function(res, req, devServer) {
      // Ensure allowedHosts is always valid
      if (devServer && devServer.options) {
        if (!devServer.options.allowedHosts || 
            !Array.isArray(devServer.options.allowedHosts) || 
            devServer.options.allowedHosts.length === 0 ||
            devServer.options.allowedHosts[0] === '') {
          devServer.options.allowedHosts = ['localhost', '127.0.0.1'];
        }
      }
      if (originalOnHeaders) {
        originalOnHeaders(res, req, devServer);
      }
    };
  }
  
  // Method 6: Use webpack's NormalModuleReplacementPlugin to patch webpack-dev-server
  const webpack = require('webpack');
  
  // Add plugin to modify webpack dev server options at runtime
  if (!config.plugins) {
    config.plugins = [];
  }
  
  // Create a custom plugin to fix allowedHosts after webpack dev server initialization
  config.plugins.push({
    apply: (compiler) => {
      compiler.hooks.afterEnvironment.tap('FixAllowedHosts', () => {
        // This runs after webpack environment is set up
        if (process.env.WEBPACK_DEV_SERVER) {
          process.env.WEBPACK_DEV_SERVER_ALLOWED_HOSTS = 'localhost,127.0.0.1';
        }
      });
    }
  });
  
  // Method 7: Suppress webpack validation warnings
  config.ignoreWarnings = [
    ...(config.ignoreWarnings || []),
    /allowedHosts/,
    /Invalid options object/
  ];
  
  return config;
};
