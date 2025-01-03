const { spawn } = require('child_process');
const path = require('path');

exports.handler = async (event, context) => {
  // Only allow POST requests
  if (event.httpMethod !== 'POST' && event.httpMethod !== 'OPTIONS') {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method not allowed' }),
    };
  }

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
      },
    };
  }

  return new Promise((resolve, reject) => {
    try {
      const pythonProcess = spawn('python3', [
        path.join(__dirname, 'recommend.py')
      ]);

      let outputData = '';
      let errorData = '';

      // Send input data to Python script
      pythonProcess.stdin.write(JSON.stringify({
        body: event.body,
        httpMethod: event.httpMethod
      }));
      pythonProcess.stdin.end();

      // Collect output data
      pythonProcess.stdout.on('data', (data) => {
        outputData += data.toString();
      });

      // Collect error data
      pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
      });

      // Handle process completion
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Python process error:', errorData);
          resolve({
            statusCode: 500,
            body: JSON.stringify({ 
              error: 'Internal server error',
              details: errorData
            })
          });
          return;
        }

        try {
          const result = JSON.parse(outputData);
          resolve({
            statusCode: 200,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*',
            },
            body: JSON.stringify(result)
          });
        } catch (e) {
          console.error('Failed to parse Python output:', outputData);
          resolve({
            statusCode: 500,
            body: JSON.stringify({ 
              error: 'Invalid response from model',
              details: e.message 
            })
          });
        }
      });

      // Handle process errors
      pythonProcess.on('error', (error) => {
        console.error('Failed to start Python process:', error);
        resolve({
          statusCode: 500,
          body: JSON.stringify({ 
            error: 'Failed to start Python process',
            details: error.message 
          })
        });
      });

    } catch (error) {
      console.error('Function error:', error);
      resolve({
        statusCode: 500,
        body: JSON.stringify({ 
          error: 'Server error',
          details: error.message 
        })
      });
    }
  });
};
