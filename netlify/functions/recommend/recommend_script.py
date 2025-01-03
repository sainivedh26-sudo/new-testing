# netlify/functions/recommend/recommend_script.py
import sys
import json
import requests
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def process_input():
    # Read input from stdin
    input_data = sys.stdin.read()
    return json.loads(input_data)

def main():
    try:
        # Get input data
        input_data = process_input()
        
        if not input_data or 'questions' not in input_data:
            print(json.dumps({
                'error': 'Please provide questions in the request body'
            }))
            sys.exit(1)

        input_questions = input_data['questions']
        num_recommendations = input_data.get('num_recommendations', 8)

        # Download model
        MODEL_URL = "https://storage.googleapis.com/model-host/model_prod.pkl"
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file.flush()
            
            # Load model
            model_data = joblib.load(tmp_file.name)

        # Clean up temp file
        os.unlink(tmp_file.name)

        # Process recommendations using TF-IDF
        vectorizer = TfidfVectorizer()
        question_texts = [q.get('text', '') for q in model_data]
        question_vectors = vectorizer.fit_transform(question_texts)
        input_vectors = vectorizer.transform(input_questions)
        
        # Calculate similarities
        similarities = cosine_similarity(input_vectors, question_vectors)
        avg_similarities = np.mean(similarities, axis=0)
        
        # Get top recommendations
        top_indices = np.argsort(avg_similarities)[-num_recommendations:][::-1]
        
        recommendations = [
            (model_data[idx].get('id', ''), float(avg_similarities[idx]))
            for idx in top_indices
        ]

        # Print results as JSON to stdout
        print(json.dumps({
            'recommended_questions': recommendations,
            'input_questions': input_questions,
            'num_recommendations': num_recommendations
        }))

    except Exception as e:
        print(json.dumps({
            'error': str(e)
        }))
        sys.exit(1)

if __name__ == '__main__':
    main()
