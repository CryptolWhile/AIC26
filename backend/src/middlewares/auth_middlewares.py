from flask import request, g
from functools import wraps
from flask import jsonify
import jwt
from src.core.config import Settings

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            secret_key = __import__('os').getenv("SECRET_KEY", "dev")
            decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
            g.user_id = decoded.get('id')  
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return decorated