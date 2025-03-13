import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import time

# Load the tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the full quantized model
quantized_model = torch.load("quantized_bert_full.pt", weights_only=False)
quantized_model.to("cpu")
quantized_model.eval()

# Function to preprocess and batch inputs dynamically
def preprocess_texts(texts, max_length=128):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return inputs

# Inference with dynamic batching
def inference_with_dynamic_batching(texts, batch_size=4):
    start_time = time.time()
    all_predictions = []

    # Split texts into batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = preprocess_texts(batch_texts)

        with torch.no_grad():
            outputs = quantized_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).tolist()
            all_predictions.extend(predictions)

    end_time = time.time()
    return all_predictions, end_time - start_time

# Test the script independently
if __name__ == "__main__":
    sample_texts = [
        "This is a positive sentence.",
        "I hate this movie so much.",
        "The weather is nice today.",
        "This project is amazing!"
    ]
    predictions, latency = inference_with_dynamic_batching(sample_texts)
    print(f"Predictions: {predictions}")
    print(f"Latency: {latency:.4f} seconds")