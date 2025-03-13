from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
from dynamic_batching import preprocess_texts, inference_with_dynamic_batching

app = Flask(__name__)

# Load the tokenizer and quantized model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
quantized_model = torch.load("quantized_bert_full.pt", weights_only=False)
quantized_model.to("cpu")
quantized_model.eval()

# Root endpoint for user-friendly message
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the BERT Inference Server!",
        "instructions": "Send a POST request to /predict with a JSON body containing 'texts' (list of strings)."
    })

# Favicon endpoint to suppress 404 errors from browsers
@app.route("/favicon.ico")
def favicon():
    return "", 204  # Return an empty response with "No Content" status

# Predict endpoint for inference
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    # Run inference with dynamic batching
    predictions, latency = inference_with_dynamic_batching(texts)
    return jsonify({
        "predictions": predictions,
        "latency": latency
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)