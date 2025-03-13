import torch
from transformers import BertForSequenceClassification

# Load the pre-trained model
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name)

# Move model to CPU for quantization (dynamic quantization works best on CPU)
model.to("cpu")
model.eval()

# Apply dynamic quantization (quantizes linear layers to INT8)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the entire quantized model
torch.save(quantized_model, "quantized_bert_full.pt")
print("Full quantized model saved as quantized_bert_full.pt")

# Optionally save the state_dict separately for reference
torch.save(quantized_model.state_dict(), "quantized_bert_state_dict.pt")
print("Quantized model state_dict saved as quantized_bert_state_dict.pt")