import argparse
import torch
import gradio
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Parse argument
parser = argparse.ArgumentParser(description="Spam Classifier gradio")
parser.add_argument(
    "--model",
    type=str,
    default="../models/distilbert-base-uncased",
    help="Model checkpoint directory"
)
args = parser.parse_args()

MODEL_PATH = args.model

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Prediction function
def predict(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).tolist()[0]
    
    return {"Ham": probs[0], "Spam": probs[1]}

# Gradio Interface
demo = gradio.Interface(
    fn=predict,
    inputs=gradio.Textbox(lines=3, placeholder="Your sms"),
    outputs=gradio.Label(num_top_classes=2),
    title=f"Spam Classifier - {MODEL_PATH}"
)

if __name__ == "__main__":
    demo.launch()
