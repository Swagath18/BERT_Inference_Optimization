import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Move model to CPU (consistent with quantization and inference setup)
model.to("cpu")
model.eval()

# Example input
text = "This is a sample sentence to test BERT inference."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Run a forward pass to test the model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

print(f"Input text: {text}")
print(f"Predicted class: {predicted_class}")