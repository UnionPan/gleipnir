// Dev server only (`react-scripts start`; ignored by production builds).
// Forwards /api to the local backend so the browser talks to one origin: the
// page works from another machine (LAN or SSH tunnel) with a single port and
// no CORS. Production builds call REACT_APP_API_URL directly.
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function setupProxy(app) {
  const apiPort = process.env.GLEIPNIR_API_PORT || '8000';
  app.use(
    createProxyMiddleware('/api', {
      target: `http://127.0.0.1:${apiPort}`,
      changeOrigin: true,
    })
  );
};
