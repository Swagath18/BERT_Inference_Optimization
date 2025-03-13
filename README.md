
# BERT Inference Optimization

This project demonstrates how to optimize a BERT model for faster inference using **post-training quantization** and **dynamic batching**, and deploy it using a Flask-based inference server.

## Features
- **Pre-trained Model Testing**: Loads and tests a pre-trained BERT model.
- **Post-Training Quantization**: Reduces model size and speeds up inference using INT8 precision.
- **Dynamic Batching**: Efficiently handles variable-length inputs during inference.
- **Flask Inference Server**: Serves the optimized model over HTTP for real-time predictions.
- **Performance Benchmarking**: Compares latency and memory usage before and after optimization.

## Requirements
- Python 3.9+
- Conda environment with the following libraries:
  ```bash
  conda create -n bert-opt python=3.9
  conda activate bert-opt
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  pip install transformers numpy onnxruntime psutil flask requests
  ```

## Structure
- `load_model.py`: Loads and tests the pre-trained BERT model.
- `quantize_model.py`: Applies post-training quantization to BERT.
- `dynamic_batching.py`: Implements dynamic batching for inference.
- `inference_server.py`: Runs a Flask server to serve the model.
- `benchmark.py`: Benchmarks performance of original vs. quantized models.
- `test_inference_server.py`: Tests the Flask server.

## How to Run
1. **Test the Pre-trained Model**:
   ```bash
   python load_model.py
   ```
   This loads the pre-trained `bert-base-uncased` model and tests it on a sample input.

2. **Quantize the Model**:
   ```bash
   python quantize_model.py
   ```
   This creates `quantized_bert_full.pt` and `quantized_bert_state_dict.pt`.

3. **Test Dynamic Batching**:
   ```bash
   python dynamic_batching.py
   ```
   This runs inference on sample texts and prints predictions and latency.

4. **Run the Inference Server**:
   ```bash
   python inference_server.py
   ```
   The server runs on `http://127.0.0.1:5000`.

5. **Test the Server**:
   ```bash
   python test_inference_server.py
   ```
   Alternatively, use `curl`:
   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"texts": ["This is great!", "I dislike this."]}'
   ```

6. **Benchmark Performance**:
   ```bash
   python benchmark.py
   ```
   This compares latency and memory usage of the original and quantized models.

## Results
- **Original Model**: Latency: 2.3208s, Memory Usage: 358.74MB (for 200 inputs)
- **Quantized Model**: Latency: 0.8008s, Memory Usage: 7.45MB (for 200 inputs)
- **Analysis**:
  - Latency improved by ~65% (2.3208s to 0.8008s) after quantization, showing scalability for larger workloads.
  - Memory usage reduced by ~98% (358.74MB to 7.45MB), making the model highly efficient for deployment.
  - Flask server processed two inputs in 0.0629s, demonstrating fast real-time inference.

## Notes
- The model (`bert-base-uncased`) is not fine-tuned, so predictions may not be meaningful for specific tasks. Fine-tuning can be added for tasks like sentiment analysis.
- The `TypedStorage` warning from PyTorch is a deprecation notice and can be ignored for now.
- The reported memory usage reflects runtime incremental usage; actual model size on disk is larger but still significantly reduced compared to the original.

## Future Improvements
- Fine-tune the model for a specific task (e.g., sentiment analysis) to make predictions meaningful.
- Add ONNX Runtime for further inference speedup.
- Deploy on edge devices (e.g., Raspberry Pi) to demonstrate real-world applicability.
- Investigate memory usage measurement to confirm runtime vs. on-disk size differences.
"# BERT_Inference_Optimization" 
