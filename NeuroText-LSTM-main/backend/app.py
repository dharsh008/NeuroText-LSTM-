from flask import Flask, request, jsonify
from flask_cors import CORS
from generate import load_checkpoint, sample
import os

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists("model.pth"):
            model = load_checkpoint("model.pth")
        else:
            print("Model not found. Please train the model.")
    return model

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', 'The')
    length = int(data.get('length', 100))
    temperature = float(data.get('temperature', 0.5))
    top_k = int(data.get('top_k', 5))
    
    net = get_model()
    if net is None:
        return jsonify({"error": "Model not found. Train it first by running train.py."}), 500
        
    try:
        generated_text = sample(net, length, prime=prompt, top_k=top_k)
        return jsonify({"text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "ready" if os.path.exists("model.pth") else "need_training"})

if __name__ == '__main__':
    app.run(port=5000)
