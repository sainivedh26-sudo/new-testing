// netlify/functions/recommend/recommend.js

const { TfIdf } = require('natural');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const cosineSimilarity = require('cosine-similarity');

class QuestionRecommender {
  constructor(questionsData) {
    this.questions = questionsData;
    this.tfidf = new TfIdf();
    
    // Create TF-IDF vectors for questions
    this.questions.forEach(q => {
      this.tfidf.addDocument(q.text || '');
    });
  }

  recommendQuestions(inputQuestions, numRecommendations = 8) {
    // Convert input questions to TF-IDF vectors
    const inputVectors = inputQuestions.map(q => {
      const vector = {};
      this.tfidf.tfidfs(q, (i, measure) => {
        vector[i] = measure;
      });
      return vector;
    });

    // Calculate similarities
    const similarities = this.questions.map((_, docIdx) => {
      const docVector = {};
      this.tfidf.tfidfs(docIdx, (i, measure) => {
        docVector[i] = measure;
      });

      // Calculate average similarity across all input questions
      const avgSimilarity = inputVectors.reduce((sum, inputVec) => {
        return sum + cosineSimilarity(Object.values(inputVec), Object.values(docVector));
      }, 0) / inputVectors.length;

      return [docIdx, avgSimilarity];
    });

    // Sort by similarity and get top recommendations
    const topRecommendations = similarities
      .sort((a, b) => b[1] - a[1])
      .slice(0, numRecommendations)
      .map(([idx, score]) => [
        this.questions[idx].id,
        score
      ]);

    return topRecommendations;
  }
}

let recommenderInstance = null;

exports.handler = async (event, context) => {
  // Handle CORS
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
    // Parse request body
    const body = JSON.parse(event.body);
    
    if (!body || !body.questions) {
      return {
        statusCode: 400,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ error: 'Please provide questions in the request body' })
      };
    }

    const inputQuestions = body.questions;
    const numRecommendations = body.num_recommendations || 8;

    // Initialize recommender if not already done
    if (!recommenderInstance) {
      // Fetch model data from cloud storage
      const modelResponse = await axios.get('https://storage.googleapis.com/model-host/model_prod.json');
      recommenderInstance = new QuestionRecommender(modelResponse.data);
    }

    // Get recommendations
    const recommendations = recommenderInstance.recommendQuestions(
      inputQuestions,
      numRecommendations
    );

    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        recommended_questions: recommendations,
        input_questions: inputQuestions,
        num_recommendations: numRecommendations
      })
    };

  } catch (error) {
    console.error('Error:', error);
    return {
      statusCode: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        error: 'Internal server error',
        details: error.message
      })
    };
  }
};
