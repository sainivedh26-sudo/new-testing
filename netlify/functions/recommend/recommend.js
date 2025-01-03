// netlify/functions/recommend/recommend.js
const { spawnSync } = require('child_process');
const path = require('path');

exports.handler = async (event, context) => {
  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
      }
    };
  }

  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    // Try different Python commands
    const pythonCommands = ['python', 'python3', 'python3.9'];
    let pythonProcess = null;
    let error = null;

    for (const cmd of pythonCommands) {
      try {
        pythonProcess = spawnSync(cmd, [path.join(__dirname, 'recommend_script.py')], {
          input: event.body,
          encoding: 'utf-8',
          stdio: ['pipe', 'pipe', 'pipe']
        });

        if (pythonProcess.status === 0) {
          break;
        }
      } catch (e) {
        error = e;
        continue;
      }
    }

    if (!pythonProcess || pythonProcess.status !== 0) {
      console.error('Python execution failed:', error || pythonProcess?.stderr);
      return {
        statusCode: 500,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          error: 'Failed to execute Python script',
          details: (error || pythonProcess?.stderr)?.toString()
        })
      };
    }

    try {
      const result = JSON.parse(pythonProcess.stdout);
      return {
        statusCode: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(result)
      };
    } catch (e) {
      return {
        statusCode: 500,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          error: 'Invalid response from model',
          details: e.message,
          output: pythonProcess.stdout
        })
      };
    }
  } catch (error) {
    console.error('Function error:', error);
    return {
      statusCode: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        error: 'Server error',
        details: error.message
      })
    };
  }
};
