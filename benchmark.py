import torch
import time
import psutil
import os
from transformers import BertTokenizer, BertForSequenceClassification
from dynamic_batching import preprocess_texts

# Load models
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Original model
original_model = BertForSequenceClassification.from_pretrained(model_name)
original_model.to("cpu")
original_model.eval()

# Quantized model
quantized_model = torch.load("quantized_bert_full.pt", weights_only=False)
quantized_model.to("cpu")
quantized_model.eval()

# Test inputs
texts = ["This is a test sentence."] * 200

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Memory in MB

# Inference function for benchmarking
def benchmark_model(model, texts):
    inputs = preprocess_texts(texts)
    start_time = time.time()
    start_memory = get_memory_usage()

    with torch.no_grad():
        outputs = model(**inputs)
        _ = outputs.logits

    end_time = time.time()
    end_memory = get_memory_usage()
    return end_time - start_time, end_memory - start_memory

# Run benchmarks
original_latency, original_memory = benchmark_model(original_model, texts)
quantized_latency, quantized_memory = benchmark_model(quantized_model, texts)

print(f"Original Model - Latency: {original_latency:.4f}s, Memory Usage: {original_memory:.2f}MB")
print(f"Quantized Model - Latency: {quantized_latency:.4f}s, Memory Usage: {quantized_memory:.2f}MB")