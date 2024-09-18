import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr

def answer_question(question, model, tokenizer, max_len=512):
    inputs = tokenizer.encode("question: " + question, return_tensors="pt", max_length=max_len, truncation=True)
    inputs = inputs.to(model.device)

    summary_ids = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load the pre-trained model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define Gradio interface
interface = gr.Interface(
    fn=lambda question: answer_question(question, model, tokenizer),
    inputs="text",
    outputs="text",
    title="Financial Question Answering System"
)

# Launch the interface
interface.launch()
