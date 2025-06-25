const { spawn } = require('child_process');

exports.handler = async (event, context) => {
  // Start Streamlit server
  const streamlit = spawn('streamlit', [
    'run',
    'app.py',
    '--server.port',
    process.env.PORT,
    '--server.address',
    '0.0.0.0'
  ]);

  // Return response while Streamlit runs in background
  return {
    statusCode: 200,
    body: 'Streamlit app is running...'
  };
};