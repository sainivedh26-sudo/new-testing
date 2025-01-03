
from flask import Flask, request, jsonify
from recommender import QuestionRecommender
import joblib

app = Flask(__name__)

# Load the model when the application starts
try:
    with open("model_prod.pkl", "rb") as model_file:
        recommender = joblib.load(model_file) 
except Exception as e:
    print(f"Error loading model: {str(e)}")
    recommender = None

@app.route('/recommend', methods=['POST'])
def recommend_questions():
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'questions' not in data:
            return jsonify({
                'error': 'Please provide questions in the request body'
            }), 400
            
        input_questions = data['questions']
        num_recommendations = data.get('num_recommendations', 8)  # Default to 8 if not specified
        
        # Validate input
        if not isinstance(input_questions, list):
            return jsonify({
                'error': 'Questions should be provided as a list'
            }), 400
            
        # Make predictions
        predictions = recommender.recommend_questions(input_questions, num_recommendations)
        
        # Return results
        return jsonify({
            'recommended_questions': predictions,
            'input_questions': input_questions,
            'num_recommendations': num_recommendations
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
