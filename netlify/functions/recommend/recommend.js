// netlify/functions/recommend/recommend.js

const { spawn } = require('child_process');
const path = require('path');

exports.handler = async (event, context) => {
  // Only allow POST requests
  if (event.httpMethod !== 'POST' && event.httpMethod !== 'OPTIONS') {
    return {
      statusCode: 405,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
      },
      body: ''
    };
  }

  // Use system Python
  const pythonProcess = spawn('/var/lang/bin/python3.9', [
    path.join(__dirname, 'recommend_script.py')
  ]);

  return new Promise((resolve, reject) => {
    let dataString = '';
    let errorString = '';

    // Send data to Python script
    pythonProcess.stdin.write(event.body);
    pythonProcess.stdin.end();

    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python process error:', errorString);
        resolve({
          statusCode: 500,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            error: 'Internal server error',
            details: errorString
          })
        });
        return;
      }

      try {
        const result = JSON.parse(dataString);
        resolve({
          statusCode: 200,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(result)
        });
      } catch (e) {
        resolve({
          statusCode: 500,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            error: 'Invalid response from model',
            details: e.message
          })
        });
      }
    });

    pythonProcess.on('error', (err) => {
      console.error('Failed to start Python process:', err);
      resolve({
        statusCode: 500,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          error: 'Failed to start Python process',
          details: err.message
        })
      });
    });
  });
};
