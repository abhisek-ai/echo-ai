"""
Flask application for EchoAI model serving on Cloud Run
"""

import os
import json
import pickle
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from flask import Flask, request, jsonify, Response
from google.cloud import storage
from google.cloud import logging as cloud_logging
import traceback

# Initialize Flask app
app = Flask(__name__)

# Set up logging
if os.getenv('GAE_ENV', '').startswith('standard'):
    # Production - use Cloud Logging
    client = cloud_logging.Client()
    client.setup_logging()
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
MODEL = None
MODEL_VERSION = os.getenv('MODEL_VERSION', 'latest')
PROJECT_ID = os.getenv('PROJECT_ID', 'your-project-id')

# Performance metrics
METRICS = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'average_latency': 0,
    'model_loaded_at': None
}


def load_model():
    """Load model from local file or GCS"""
    global MODEL, METRICS
    
    try:
        model_path = os.getenv('MODEL_PATH', '/app/models/best_model.pkl')
        
        # Try loading from local file first
        if os.path.exists(model_path):
            logger.info(f"Loading model from local path: {model_path}")
            with open(model_path, 'rb') as f:
                MODEL = pickle.load(f)
        else:
            # Load from GCS if local file doesn't exist
            logger.info("Loading model from GCS")
            bucket_name = os.getenv('MODEL_BUCKET', f'{PROJECT_ID}-models')
            blob_name = os.getenv('MODEL_BLOB', 'models/best_model.pkl')
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            model_bytes = blob.download_as_bytes()
            MODEL = pickle.loads(model_bytes)
        
        METRICS['model_loaded_at'] = datetime.now().isoformat()
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


@app.before_first_request
def initialize():
    """Initialize model on first request"""
    load_model()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    health_status = {
        'status': 'healthy' if MODEL is not None else 'unhealthy',
        'model_loaded': MODEL is not None,
        'model_version': MODEL_VERSION,
        'timestamp': datetime.now().isoformat()
    }
    
    status_code = 200 if MODEL is not None else 503
    return jsonify(health_status), status_code


@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check endpoint"""
    if MODEL is not None:
        return jsonify({'ready': True}), 200
    else:
        return jsonify({'ready': False}), 503


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return service metrics"""
    return jsonify(METRICS), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    global METRICS
    
    start_time = time.time()
    METRICS['total_requests'] += 1
    
    try:
        # Validate model is loaded
        if MODEL is None:
            logger.warning("Model not loaded, attempting to load...")
            if not load_model():
                return jsonify({'error': 'Model not available'}), 503
        
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['text']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Prepare features for prediction
        features = prepare_features(data)
        
        # Make prediction
        prediction = MODEL.predict(features)[0]
        
        # Generate confidence scores (if model supports predict_proba)
        confidence = None
        try:
            probabilities = MODEL.predict_proba(features)[0]
            confidence = float(max(probabilities))
        except:
            pass
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # in ms
        
        # Update metrics
        METRICS['successful_predictions'] += 1
        update_average_latency(latency)
        
        # Prepare response
        response = {
            'prediction': str(prediction),
            'confidence': confidence,
            'model_version': MODEL_VERSION,
            'latency_ms': round(latency, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log prediction for monitoring
        logger.info(f"Prediction made: {prediction}, latency: {latency:.2f}ms")
        
        return jsonify(response), 200
        
    except Exception as e:
        METRICS['failed_predictions'] += 1
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple inputs"""
    global METRICS
    
    start_time = time.time()
    
    try:
        if MODEL is None:
            if not load_model():
                return jsonify({'error': 'Model not available'}), 503
        
        data = request.get_json()
        if not data or 'instances' not in data:
            return jsonify({'error': 'No instances provided'}), 400
        
        instances = data['instances']
        predictions = []
        
        for instance in instances:
            features = prepare_features(instance)
            prediction = MODEL.predict(features)[0]
            predictions.append(str(prediction))
        
        latency = (time.time() - start_time) * 1000
        
        response = {
            'predictions': predictions,
            'count': len(predictions),
            'model_version': MODEL_VERSION,
            'latency_ms': round(latency, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        METRICS['successful_predictions'] += len(predictions)
        update_average_latency(latency)
        
        return jsonify(response), 200
        
    except Exception as e:
        METRICS['failed_predictions'] += 1
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'message': str(e)}), 500


@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload model from storage (useful for model updates)"""
    try:
        if load_model():
            return jsonify({
                'status': 'success',
                'message': 'Model reloaded successfully',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to reload model'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    info = {
        'model_loaded': MODEL is not None,
        'model_version': MODEL_VERSION,
        'model_type': type(MODEL).__name__ if MODEL else None,
        'model_loaded_at': METRICS['model_loaded_at'],
        'total_predictions': METRICS['successful_predictions'],
        'project_id': PROJECT_ID
    }
    return jsonify(info), 200


def prepare_features(data: Dict[str, Any]) -> np.ndarray:
    """
    Prepare features from input data for model prediction
    
    Args:
        data: Input data dictionary
        
    Returns:
        Feature array for model input
    """
    # Extract text features
    text = data.get('text', '')
    rating = data.get('rating', 0)
    
    # Basic feature engineering (customize based on your model)
    features = []
    
    # Text length
    features.append(len(text))
    
    # Word count
    features.append(len(text.split()))
    
    # Rating
    features.append(float(rating))
    
    # Sentiment indicators (basic)
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love']
    negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible', 'hate', 'worst']
    
    positive_count = sum(1 for word in positive_words if word in text.lower())
    negative_count = sum(1 for word in negative_words if word in text.lower())
    
    features.append(positive_count)
    features.append(negative_count)
    
    # Add more features as needed based on your model training
    
    return np.array([features])


def update_average_latency(new_latency: float):
    """Update running average of latency"""
    global METRICS
    
    current_avg = METRICS['average_latency']
    count = METRICS['successful_predictions']
    
    if count == 1:
        METRICS['average_latency'] = new_latency
    else:
        METRICS['average_latency'] = ((current_avg * (count - 1)) + new_latency) / count


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)