import json
import os
import requests
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

class SimpleRecommender:
    def __init__(self, questions_data):
        self.questions = questions_data
        self.vectorizer = TfidfVectorizer()
        # Create vectors for questions
        question_texts = [q.get('text', '') for q in questions_data]
        self.question_vectors = self.vectorizer.fit_transform(question_texts)
        
    def recommend_questions(self, input_questions, num_recommendations=8):
        # Convert input questions to vectors
        input_vectors = self.vectorizer.transform(input_questions)
        
        # Calculate similarities
        similarities = cosine_similarity(input_vectors, self.question_vectors)
        
        # Get average similarity scores
        avg_similarities = np.mean(similarities, axis=0)
        
        # Get top recommendations
        top_indices = np.argsort(avg_similarities)[-num_recommendations:][::-1]
        
        # Return recommendations with scores
        recommendations = []
        for idx in top_indices:
            recommendations.append((
                self.questions[idx].get('id', ''),
                float(avg_similarities[idx])
            ))
        
        return recommendations

def handler(event, context):
    """Netlify function handler for question recommendations"""
    # Handle CORS
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': ''
        }

    try:
        # Parse request
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        
        if not body or 'questions' not in body:
            return {
                'statusCode': 400,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Please provide questions'})
            }
            
        input_questions = body['questions']
        num_recommendations = body.get('num_recommendations', 8)

        # Download and load model
        MODEL_URL = "https://storage.googleapis.com/model-host/model_prod.pkl"
        
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file.flush()
                
                questions_data = joblib.load(tmp_file.name)
            
            os.unlink(tmp_file.name)
            
            # Initialize recommender and get predictions
            recommender = SimpleRecommender(questions_data)
            predictions = recommender.recommend_questions(input_questions, num_recommendations)
            
            return {
                'statusCode': 200,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'recommended_questions': predictions,
                    'input_questions': input_questions,
                    'num_recommendations': num_recommendations
                })
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': f'Error: {str(e)}'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }
